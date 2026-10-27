"""Diagnostic range-integrated layer means and a deterministic MSC effective-path approximation (JAX).

The "g4" label does not imply transport equivalence or validated physical endpoints. Per lane:
  * effective path t from geometric z via z = lambda1 (1 - exp(-t / lambda1)), using a mean energy and two
    fixed-point passes; z >= lambda1 receives a forced-stop sentinel, not a particle-level traversal theorem;
  * t in n equal slices; each slice's mean loss is T - R^-1(R(T) - dl) with a native CSDA range, then Urban
    fluctuations are sampled about that mean; dl >= R(T) bypasses fluctuations. Endpoint/cut approximations
    remain different from a full G4 step, and this inclusive mean does not certify hard-loss tails;
  * the layer stops the muon when its kinetic energy ends below T_STOP (0.1 MeV), as the legacy model.
Tables: release-matched Geant4 aluminium tables for each charge (beamfastsim.g4tables: the StoppingTarget_Al export
by default, the G4_Al slab export for single-foil closure); materials without a shipped
table (IPA polyethylene) use the range of the fast Bethe formula with constant stopping power below 0.1 MeV and no
path-length correction.
"""
from typing import NamedTuple

import numpy as np

import jax.numpy as jnp
from scipy.integrate import cumulative_trapezoid

from mu2efastsim import eloss, eloss_jax

from . import g4tables

T_STOP = 0.1


class Tables(NamedTuple):
    """Log-log range table (from zero energy) and msc transport mean free path, MeV and mm."""
    loge: object
    logr: object
    lam_loge: object
    lam_log: object


def _tables(energy, rng, lam_e=None, lam=None):
    e, r = np.asarray(energy, float), np.asarray(rng, float)
    if np.any(np.diff(e) <= 0) or np.any(np.diff(r) <= 0) or r[0] <= 0:
        raise ValueError("increasing positive range table required")
    if lam is None:
        lam_e, lam = e[[0, -1]], np.full(2, 1e300)
    return Tables(*(jnp.asarray(np.log(v)) for v in (e, r, lam_e, lam)))


def g4_tables(charge, table="target"):
    """Range/msc tables of one charge from the shipped export `table` ("target" StoppingTarget_Al, "slab" G4_Al)."""
    material = {"target": "StoppingTarget_Al", "slab": "G4_Al"}[table]
    t = g4tables.load(g4tables.PATHS[table], expected_material=material)
    return _tables(t["energy"], t[f"csda_range_{charge}"], t["msc_energy"], t["msc_lambda1"])


def fast_tables(material, tcut, emin=0.1, emax=1000.0, per_decade=256):
    """Range of the fast restricted Bethe formula; constant dE/dx below emin (no msc path correction)."""
    e = np.geomspace(emin, emax, int(round(np.log10(emax / emin) * per_decade)) + 1)
    p = np.sqrt(e * (e + 2 * eloss.MMU))
    s = np.asarray(eloss.dedx_restricted(material, p, tcut))
    r = emin / s[0] + cumulative_trapezoid(1 / s, e, initial=0)
    return _tables(e, r)


def range_of(tb, t):
    """CSDA range [mm] of kinetic energy t [MeV]; linear in t below the first node."""
    lt = jnp.log(jnp.maximum(t, 1e-300))
    e0, r0 = tb.loge[0], tb.logr[0]
    return jnp.where(lt < e0, jnp.exp(r0 + lt - e0), jnp.exp(jnp.interp(lt, tb.loge, tb.logr)))


def energy_of(tb, r):
    """Inverse of range_of."""
    lr = jnp.log(jnp.maximum(r, 1e-300))
    e0, r0 = tb.loge[0], tb.logr[0]
    return jnp.where(lr < r0, jnp.exp(e0 + lr - r0), jnp.exp(jnp.interp(lr, tb.logr, tb.loge)))


def lambda1(tb, t):
    return jnp.exp(jnp.interp(jnp.log(jnp.maximum(t, 1e-300)), tb.lam_loge, tb.lam_log))


def pick(fn, tbs, kind, *args):
    """fn(tables, *args) for the per-lane table index `kind` (all tables evaluated, then selected)."""
    out = fn(tbs[0], *args)
    for k in range(1, len(tbs)):
        out = jnp.where(kind == k, fn(tbs[k], *args), out)
    return out


def kinetic(p):
    return jnp.sqrt(p * p + eloss.MMU ** 2) - eloss.MMU


def momentum(t):
    t = jnp.maximum(t, 0.0)
    return jnp.sqrt(t * (t + 2 * eloss.MMU))


def true_path(tbs, kind, t0, z, passes=2):
    """Effective-path surrogate; z/lambda >= 1 gets a forced-stop sentinel, not a measured true path."""
    r0 = pick(range_of, tbs, kind, t0)
    t = z
    for _ in range(passes):
        t1 = pick(energy_of, tbs, kind, jnp.maximum(r0 - t, 0.0))
        lam = pick(lambda1, tbs, kind, 0.5 * (t0 + jnp.where(t < r0, t1, 0.0)))
        x = z / lam
        t = jnp.where(x < 1, -lam * jnp.log1p(-jnp.minimum(x, 1 - 1e-15)), 2 * r0 + 1)
    return t


def slices(key, tbs, kind, mc, p, length, n_slices, tcut, fold_in):
    """Range-mean Urban slices over `length` [mm]; returns the outgoing kinetic energy (0 on full stopping)."""
    t = kinetic(p)
    dl = length / n_slices
    for s in range(n_slices):
        r = pick(range_of, tbs, kind, t)
        full = dl >= r
        mean = jnp.maximum(t - pick(energy_of, tbs, kind, jnp.maximum(r - dl, 0.0)), 0.0)
        loss = eloss_jax._sample(fold_in(key, s), mc, momentum(t), jnp.where(full, 1.0, dl), tcut,
                                 jnp.asarray(eloss.MMU), mean=jnp.where(full, 0.0, mean))
        t = jnp.where(full | (t <= 0), 0.0, jnp.maximum(t - loss, 0.0))
    return t
