"""Readers for the stock calorimeter inputs and outputs used in the fast-vs-stock comparisons."""
from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from .pulse import PulseCache
from .readout import PropagationTable


def load_steps(path, sim_crystal=None, **timing_kw):
    """CaloShowerSteps and per-event timing as ``(events, timing, steps)``.

    ``path`` is either a stock art file (read with :mod:`calofastsim.artsteps`; needs ``sim_crystal`` for LRU and
    yields, and takes the on-spill EWM settings in ``timing_kw``) or a ROOT table export with this schema:

    - ``response/events``, one row per event: ``event`` (art event number), ``pbtime`` (ProtonBunchTimeMC, ns),
      ``mbtime`` (nominal DR period), ``event_length``, ``spill`` (1 = on spill), ``marker``
      (timeFromProtonsToDRMarker), ``crystal_length`` (mm);
    - ``response/steps``, one row per CaloShowerStep in collection order: ``event``, ``crystal`` (volumeG4ID),
      ``energy_g4``, ``energy_birks`` (MeV), ``time`` (ns, double), ``z`` (crystal-frame depth, mm),
      ``lru``, ``yield0``, ``yield1`` (PE/MeV of the two SiPMs).

    Events are returned sorted by event number; ``steps["event"]`` indexes them.
    """
    if str(path).endswith(".art"):
        from .artsteps import steps_from_art
        if sim_crystal is None:
            raise ValueError("reading steps from art needs the Sim_crystal table")
        return steps_from_art(path, sim_crystal, **timing_kw)
    with uproot.open(path) as f:
        ev = f["response/events"].arrays(library="np")
        st = f["response/steps"].arrays(["event", "crystal", "energy_g4", "energy_birks", "time", "z", "lru",
                                         "yield0", "yield1"], library="np")
    order = np.argsort(ev["event"], kind="stable")
    events = ev["event"][order]
    idx = np.searchsorted(events, st["event"])
    if not np.array_equal(events[idx], st["event"]):
        raise ValueError("steps refer to events missing from the event table")
    timing = {"pbtime": ev["pbtime"][order], "mbtime": ev["mbtime"][order], "event_length": ev["event_length"][order],
              "onspill": ev["spill"][order] == 1, "marker": ev["marker"][order],
              "crystal_length": ev["crystal_length"][order]}
    steps = {"event": idx, "crystal": st["crystal"].astype(np.int64), "energy_g4": st["energy_g4"],
             "energy_birks": st["energy_birks"], "time": st["time"], "z_local": st["z"], "lru": st["lru"],
             "yields": np.stack([st["yield0"], st["yield1"]], axis=1)}
    return events, timing, steps


def load_sim_crystal(path):
    """``Sim_crystal.txt`` rows: crystal, LRU, PE/MeV x2, ADC/MeV x2 (float32, as CalSimParams stores them)."""
    rows = np.loadtxt(path, comments="#")
    rows = rows[np.argsort(rows[:, 0])]
    return {"lru": rows[:, 1].astype(np.float32), "pe_per_mev": rows[:, 2:4].astype(np.float32),
            "adc_per_mev": rows[:, 4:6].astype(np.float32)}


def adc_per_pe(sim, sipm):
    """``readoutScaleFactor``: ADCPerMeV / pePerMeV in float32, as double."""
    sipm = np.asarray(sipm)
    c, i = sipm // 2, sipm % 2
    return (sim["adc_per_mev"][c, i] / sim["pe_per_mev"][c, i]).astype(np.float64)


def load_pulse(path, digi_sampling=5.0, hist="h_waveform"):
    with uproot.open(path) as f:
        h = f[hist]
        edges = h.axis().edges()
        return PulseCache.from_histogram(h.values(), float(edges[0]), float(edges[-1]), digi_sampling)


def load_noise(path, hist="histo_0"):
    with uproot.open(path) as f:
        return f[hist].values().astype(np.float32).astype(np.float64)


def load_propagation(path, hist="h_zposTime"):
    with uproot.open(path) as f:
        h = f[hist]
        return PropagationTable.from_histogram(h.values(), h.axis(0).edges(), h.axis(1).edges())


def load_ntuple(path, arm, events):
    """Stock calo branches of one EventNtuple instance, aligned to ``events`` (event numbers, sorted)."""
    with uproot.open(Path(path)) as f:
        t = f[f"{arm}/ntuple"]
        a = t.arrays(["evtinfo", "calodigis", "calorecodigis", "calohits", "caloclusters"])
    num = ak.to_numpy(a["evtinfo"]["event"])
    pos = np.searchsorted(events, num)
    if np.any(pos >= len(events)) or not np.array_equal(events[np.minimum(pos, len(events) - 1)], num):
        raise ValueError("ntuple events missing from the step table")
    return pos, a
