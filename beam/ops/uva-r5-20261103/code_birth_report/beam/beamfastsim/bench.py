"""Throughput of the transport on synthetic muons in a synthetic DS-sized map (no CVMFS), per configuration.

  python3 -m beamfastsim.bench [--n N] [--reps R]
Configurations: full physics, no energy loss/MSC in layers (field stepping only), and field stepping with B = 0.
"""
import argparse
import json
import time

import numpy as np

import jax
import jax.numpy as jnp

from . import fieldmap, geometry, transport


def synthetic(n_pts=(97, 97, 521)):
    n = np.array(n_pts)
    lo = np.array([-5096.0, -1200.0, 3071.0])
    B = np.zeros((*n, 3))
    z = lo[2] + np.arange(n[2]) * 25.0
    B[..., 2] = (2.0 - 1.0 * np.clip((z - 4000) / 2000, 0, 1))[None, None, :]
    return fieldmap.FieldMap(jnp.asarray(lo), jnp.asarray(np.full(3, 25.0)), jnp.asarray(n), jnp.asarray(B))


def geom():
    nf = 37
    z = 5871 + (np.arange(nf) - 18) * 22.222222
    return geometry.BeamGeometry(-3904.0, z, np.full(nf, 21.5), np.full(nf, 75.0), np.full(nf, 0.0528),
                                 4204.6, 25.4, 235.0, 525.0, 300.0, 0.511, 6901.0, 7901.0, 454.0, 4280.0, 8530.0)


def muons(n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.stack([-3904.0 + rng.normal(0, 40, n), rng.normal(0, 40, n), np.full(n, 4179.0)], 1)
    p = rng.gamma(9.0, 5.0, n)
    ct = rng.uniform(0.5, 1.0, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - ct ** 2)
    return x, np.stack([p * st * np.cos(ph), p * st * np.sin(ph), p * ct], 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--slices", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=200)
    a = ap.parse_args(argv)
    fm, g = synthetic(), geom()
    x, p3 = muons(a.n)
    out = {"device": str(jax.devices()[0]), "n": a.n}
    for name, prm in (("full", transport.Params(n_slices=a.slices)),
                      ("no_msc", transport.Params(n_slices=a.slices, msc=False)),
                      ("no_layers", transport.Params(n_slices=0))):
        st = transport.initial_state(jax.random.PRNGKey(0), x, p3, np.zeros(a.n), -np.ones(a.n), prm)
        res, nst = transport.transport_all(fm, g, prm, jax.random.PRNGKey(1), st, chunk=a.chunk)
        t0 = time.time()
        for r in range(a.reps):
            res, nst = transport.transport_all(fm, g, prm, jax.random.PRNGKey(2 + r), st, chunk=a.chunk)
        dt = (time.time() - t0) / a.reps
        out[name] = {"muons_per_s": a.n / dt, "lane_steps": int(nst), "us_per_lane_step": 1e6 * dt / int(nst),
                     "stop_frac": float(np.mean(np.asarray(res.fate) == 1))}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
