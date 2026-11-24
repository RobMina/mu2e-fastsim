"""Device overburden transport for the design muon sample: the stage-C world loss and band-weighted Highland
scattering to the MSC plane of transport_jax.DeviceTransport (the extracted/KPP device chain), without the bar loop.

Host code pads the preselected muons into fixed-shape batches; each batch runs as one jitted program in the device float
(mu2efastsim.precision) and returns (origin, direction on the plane, |p| at the plane, survived?). RNG: threefry keys
folded by batch index, so a sample is reproducible for a fixed seed, batch size and runtime; not the host RNG stream.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mu2efastsim import precision as dprec

from .. import transport_jax as TJ

dprec.register(__name__)


class EntryTransport:
    """World constants on the device (one DeviceTransport, stage C); reuse it for every gen chunk of a sample."""

    def __init__(self, geom, world, batch_size=4096):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dev = TJ.DeviceTransport(geom, world, stage="C", batch_size=batch_size)
        self.y_plane, self.batch_size = self.dev.y, batch_size
        self.kernel = dprec.jit(self._batch)

    def _batch(self, key, o, d, p, active):
        pp, alive, xx, bh = self.dev.world(jax.random.fold_in(key, 0), o, d, p, active)
        pm = jnp.maximum(jnp.sqrt(p * jnp.maximum(pp, 1.)), 1.)
        o2, d2, _ = TJ.scatter(jax.random.fold_in(key, 1), o, d, pm, self.y_plane, xx, bh)
        return o2, d2, pp, alive & active

    def run(self, o, d, p, seed):
        """(origin [n, 3] on the plane, direction [n, 3], p [n] at the plane, alive [n]) for downward lines o + s d."""
        o, d, p = (np.asarray(a, np.float64) for a in (o, d, p))
        n = len(p)
        if n and (not np.isfinite(o).all() or not np.isfinite(d).all() or (p <= 0).any() or (d[:, 1] >= 0).any()):
            raise ValueError("entry transport needs finite downward lines with positive momentum")
        key = jax.random.key(int(seed) & 0xffffffff)
        out = [np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n), np.zeros(n, bool)]
        b = self.batch_size
        for i, start in enumerate(range(0, n, b)):
            m = min(b, n - start)
            ob = np.zeros((b, 3)); ob[:, 1] = self.y_plane + 1.0; ob[:m] = o[start:start + m]
            db = np.zeros((b, 3)); db[:, 1] = -1.0; db[:m] = d[start:start + m]
            pb = np.full(b, 1000.0); pb[:m] = p[start:start + m]
            res = dprec.host(jax.device_get(self.kernel(jax.random.fold_in(key, i), jnp.asarray(ob), jnp.asarray(db),
                                                        jnp.asarray(pb), jnp.arange(b) < m)))
            for k in range(4):
                out[k][start:start + m] = res[k][:m]
        return tuple(out)
