"""W9 (CRV module design figures of merit): the approximations of python -m crvfastsim.design evaluate.

None has a switch yet; each is listed with what stock does instead (mu2efastsim.approximations scheme).
"""
from mu2efastsim.approximations import Workflow

WORKFLOW = Workflow(
    "W9", "CRV module design figures of merit",
    (),
    fixed={
        "primary_muon_lines": "Geant4 transport of CRY muons with secondaries (straight primary lines here; "
                              "secondaries, scattering and energy loss in the CRV are not simulated)",
        "scaled_optics": "photon lookup tables made by the stock optical simulation for each counter cross-section, "
                         "fiber and length (here: the nearest stock table, scaled; fiber via the stock yield ratio)",
        "expected_pe_poisson": "stock photons, SiPM, waveforms and pulse reco on Geant4 steps (here: expected PE per "
                               "counter end, Poisson, no visible-energy fluctuations, calibrated at nominal)",
        "coincidence_rule": "stock CrvCoincidenceFinder with time, slope and distance cuts (here: k-of-N layers, "
                            "same-plus-adjacent PE threshold, the slope window and minimum cluster PE on the "
                            "most-lit counter per layer; no time or distance cuts)",
        "ce_like_proxy": "CE-like track selection on full Geant4 + reco cosmics (here: lines through a cylinder "
                         "about the DS axis)",
        "deadtime_rate_model": "stock CRV response and reco on beam-background frames in the design geometry "
                               "(here: accidental and correlated rates calibrated at the nominal design)",
    },
)
