"""crvfastsim.veto -- tier C4 of the veto-optimisation framework: a supplied veto rule applied to selected tracks and
CRV coincidences, giving the veto inefficiency (cosmics) or the deadtime (signal + beam).
The card is an input, never chosen here.

    card      VetoCard: the rule (time window around the track, coincidence time gate, PE / layer / sector cuts);
              from_mu2e_cosmic_ana(cutset) reads S. Grant's YAML cut sets, from_dict for anything else
    apply     vetoed(tracks, coincs, card) -> bool per track, fully vectorised on awkward arrays
    metrics   inefficiency / deadtime with a Jeffreys interval, per sector of the vetoing coincidence
    deadtime  methods S (each selected signal track against its own event) and F (beam frames against the selected
              signal t0 distribution), python -m crvfastsim.veto.deadtime

Inputs are flat awkward records named as the EventNtuple branches the existing analyses read: per track t0
(trksegs.time at the tracker middle, or a track time), and per event crvcoincs.{time, timeStart, timeEnd, PEs,
nLayers, sectorType}. Numpy + awkward only: importable by Mu2eBO kits and grid jobs.
"""
from .card import VetoCard
from .apply import vetoed
from .metrics import fraction, jeffreys

__all__ = ["VetoCard", "vetoed", "fraction", "jeffreys"]
