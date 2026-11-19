"""Muon states at the CRV for the design figures of merit: CRY primaries carried through the world material above
the CRV (stage C of crvfastsim.transport: straggled loss in the GDML world, Highland scattering to the MSC plane).

A design point then starts every muon on the MSC plane with its momentum there, so the overburden's energy loss and
scattering are in the sample, not in the design model; muons that range out above the plane are dropped (they reach
neither the CRV nor the detector). The CE-like proxy is applied to the transported lines.
"""
from __future__ import annotations

import numpy as np

from mu2efastsim import eloss, materials
from mu2efastsim import transport as core_tr

from .. import geometry as G
from .. import transport as TR
from . import fom as F


def crv_entry_states(gen, world_gdml, counters, rng, proxy=F.PROXY, preselect_r=2000.0):
    """dict origin [n, 3] (on the MSC plane), direction [n, 3], p [n] (MeV/c at the plane) of the CE-proxy muons of a
    CRY gen table, plus counts. Muons are preselected on their generated lines with a wider proxy (preselect_r) so
    scattering cannot move a selected muon in from outside; the proxy itself is applied after transport."""
    mu = TR.all_primary_muons(gen)
    p0 = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
    o = mu[["x", "y", "z"]].to_numpy(np.float64)
    d = mu[["px", "py", "pz"]].to_numpy(np.float64) / np.where(p0 > 0, p0, 1.0)[:, None]
    wide = F.proxy_hits(o, d, dict(proxy, r=preselect_r)) & (p0 > 0) & (d[:, 1] < 0)
    geom = G.from_offline_config(G.find_counters(counters))
    y_plane = TR.msc_plane(geom)
    world, _ = TR.load_world(world_gdml, banded=True, y_plane=y_plane)
    edges = world.edges if isinstance(world, core_tr.BandedWorld) else TR.band_edges(y_plane)
    ow, dw, pw = o[wide], d[wide], p0[wide]
    p_plane, xx0, alive, bxx0, bh = TR.world_loss(rng, ow, dw, pw, world, edges)
    p_msc = np.maximum(np.sqrt(pw * np.maximum(p_plane, 1.0)), 1.0)
    o2, d2, _ = core_tr.msc_to_plane(rng, ow, dw, p_msc, y_plane, eloss.radiation_length_mm(materials.AIR), xx0,
                                     bxx0, bh)
    keep = alive & F.proxy_hits(o2, d2, proxy)
    ev = mu[["run", "subrun", "event"]].to_numpy(np.int64)[wide][keep]
    return dict(origin=o2[keep], direction=d2[keep], p=p_plane[keep], event_key=ev, n_generated_muons=int(len(mu)),
                n_preselected=int(wide.sum()), n_ranged_out=int((~alive).sum()),
                n_selected_generated_lines=int(F.proxy_hits(o, d, proxy)[wide].sum()), y_plane=float(y_plane))
