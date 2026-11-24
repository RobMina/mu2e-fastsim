"""Muon states at the CRV for the design figures of merit: CRY primaries carried through the world material above
the CRV (stage C of crvfastsim.transport: straggled loss in the GDML world, Highland scattering to the MSC plane).

A design point then starts every muon on the MSC plane with its momentum there, so the overburden's energy loss and
scattering are in the sample, not in the design model; muons that range out above the plane are dropped (they reach
neither the CRV nor the detector). The CE-like proxy is applied to the transported lines.

backend "numpy" is the host reference; "jax" runs the overburden on the device with the stage-C kernels of
transport_jax (EntryTransport), the same world transport as the extracted (KPP) chain. Only the proxy is Run 2-specific.
"""
from __future__ import annotations

import numpy as np

from mu2efastsim import eloss, materials
from mu2efastsim import transport as core_tr

from .. import geometry as G
from .. import transport as TR
from . import fom as F


class EntryWorld:
    """Geometry, MSC plane and stage-C world of one GDML + counters file, loaded once (device kernels built lazily)."""

    def __init__(self, world_gdml, counters):
        self.geom = G.from_offline_config(G.find_counters(counters))
        self.y_plane = TR.msc_plane(self.geom)
        self.world, _ = TR.load_world(world_gdml, banded=True, y_plane=self.y_plane)
        self.edges = self.world.edges if isinstance(self.world, core_tr.BandedWorld) else TR.band_edges(self.y_plane)
        self._device = None

    def device(self, batch_size=4096):
        if self._device is None:
            from .entry_jax import EntryTransport
            self._device = EntryTransport(self.geom, self.world, batch_size=batch_size)
        return self._device


def crv_entry_states(gen, world_gdml, counters, rng, proxy=F.PROXY, preselect_r=2000.0, backend="numpy", prepared=None):
    """dict origin [n, 3] (on the MSC plane), direction [n, 3], p [n] (MeV/c at the plane) of the CE-proxy muons of a
    CRY gen table, plus counts. Muons are preselected on their generated lines with a wider proxy (preselect_r) so
    scattering cannot move a selected muon in from outside; the proxy itself is applied after transport.
    prepared: an EntryWorld of the same world_gdml/counters (reused across gen chunks); backend: "numpy" | "jax"."""
    mu = TR.all_primary_muons(gen)
    p0 = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
    o = mu[["x", "y", "z"]].to_numpy(np.float64)
    d = mu[["px", "py", "pz"]].to_numpy(np.float64) / np.where(p0 > 0, p0, 1.0)[:, None]
    wide = F.proxy_hits(o, d, dict(proxy, r=preselect_r)) & (p0 > 0) & (d[:, 1] < 0)
    ew = prepared if prepared is not None else EntryWorld(world_gdml, counters)
    ow, dw, pw = o[wide], d[wide], p0[wide]
    if backend == "jax":
        o2, d2, p_plane, alive = ew.device().run(ow, dw, pw, int(rng.integers(0, 2 ** 32, dtype=np.uint64)))
    elif backend == "numpy":
        p_plane, xx0, alive, bxx0, bh = TR.world_loss(rng, ow, dw, pw, ew.world, ew.edges)
        p_msc = np.maximum(np.sqrt(pw * np.maximum(p_plane, 1.0)), 1.0)
        o2, d2, _ = core_tr.msc_to_plane(rng, ow, dw, p_msc, ew.y_plane, eloss.radiation_length_mm(materials.AIR), xx0,
                                         bxx0, bh)
    else:
        raise ValueError(f"unknown entry-state backend: {backend}")
    keep = alive & F.proxy_hits(o2, d2, proxy)
    ev = mu[["run", "subrun", "event"]].to_numpy(np.int64)[wide][keep]
    n_events = int(len(gen[["run", "subrun", "event"]].drop_duplicates()))
    return dict(origin=o2[keep], direction=d2[keep], p=p_plane[keep], p_generated=pw[keep], event_key=ev,
                n_generated_muons=int(len(mu)), n_generated_events=n_events,
                n_preselected=int(wide.sum()), n_ranged_out=int((~alive).sum()),
                n_selected_generated_lines=int(F.proxy_hits(o, d, proxy)[wide].sum()), y_plane=float(ew.y_plane))
