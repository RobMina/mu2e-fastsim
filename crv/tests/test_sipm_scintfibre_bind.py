"""SiPM bind into scint-fibre-fastsim with a public synthetic photon map (no Mu2e inputs)."""

from __future__ import annotations

import numpy as np

from crvfastsim.response import sipm
from scintfibrefastsim.backends import Arrival, Mu2eSiPMBackend, uniform_photon_map


def test_bind_real_simulate_with_uniform_map():
    pmap = uniform_photon_map(40)
    cfg = sipm.SiPMConfig(pixel_id="truncate", thermal_rate=0.0, trap0_prob=0.0, trap1_prob=0.0)
    backend = Mu2eSiPMBackend(simulate=sipm.simulate, photon_map=pmap, config=cfg)
    assert backend.bound
    arrivals = [Arrival(time_ns=float(i), channel=i % 4, energy_eV=2.5) for i in range(50)]
    out = backend.detect(arrivals, seed=3)
    assert out["backend"] == "mu2e_sipm"
    assert out["n_charges"] > 0
    assert out["n_pe"] > 0
    assert np.asarray(out["pe"]).sum() == out["n_pe"]
