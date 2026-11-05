"""CaloShowerStep -> per-SiPM photoelectrons (CaloShowerROMaker::makeReadoutHits), vectorised over all events.

Uses the validated primitives of :mod:`calofastsim.response` and :mod:`calofastsim.propagation`. Photon statistics
and travel times use the caller's NumPy generator, so with them on only distributions can match stock.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import response
from .propagation import build_cdf, propagation_time


@dataclass(frozen=True)
class ReadoutConfig:
    digitization_start: float = 450.0
    digitization_end: float = 1795.0
    digitization_buffer: float = 10.0
    birks: bool = True
    lru: bool = True
    pe_stat: bool = True
    travel: bool = True


@dataclass(frozen=True)
class PropagationTable:
    cdf: np.ndarray
    times: np.ndarray
    dz: float

    @classmethod
    def from_histogram(cls, contents, x_edges, y_edges):
        """``contents`` is (depth, time) without flow bins, as ``CaloPhotonPropagation::buildTable`` reads it."""
        y = np.asarray(y_edges, dtype=np.float64)
        times = (0.5 * (y[:-1] + y[1:])).astype(np.float32)
        x = np.asarray(x_edges, dtype=np.float64)
        return cls(build_cdf(contents), times, np.float32(x[1] - x[0]))


def make_readouts(step_event, crystal, energy_g4, energy_birks, time, z_local, lru, yields, timing, cfg: ReadoutConfig,
                  *, table: PropagationTable | None = None, rng=None):
    """Return flat PE groups ``(event, sipm, time float32, count)``.

    ``timing`` holds per-event arrays ``pbtime, mbtime, event_length, onspill, marker, crystal_length``. With
    ``travel`` on, every PE gets its own propagation time (count 1); otherwise PEs of one step and SiPM share the
    folded step time and are returned once with their count.
    """
    ev = np.asarray(step_event, dtype=np.int64)
    get = {k: np.asarray(v)[ev] for k, v in timing.items()}
    hit, accepted = response.fold_step_time(
        np.asarray(time, dtype=np.float64), pbtime=get["pbtime"], mbtime=get["mbtime"],
        event_length=get["event_length"], onspill=get["onspill"].astype(bool),
        digitization_start=cfg.digitization_start, digitization_end=cfg.digitization_end,
        digitization_buffer=cfg.digitization_buffer, time_from_protons_to_dr_marker=get["marker"])
    a = np.flatnonzero(accepted)
    energy = response.corrected_energy(np.asarray(energy_g4)[a], np.asarray(energy_birks)[a], np.asarray(z_local)[a],
                                       get["crystal_length"][a], np.asarray(lru)[a], birks=cfg.birks,
                                       longitudinal=cfg.lru)
    mean = response.photoelectron_mean(energy, np.asarray(yields)[a])
    if cfg.pe_stat:
        count = rng.poisson(mean.astype(np.float64)).astype(np.int64)
    else:
        count = response.nonnegative_lround(mean).astype(np.int64)
    step = np.repeat(a, 2)
    sipm = 2 * np.asarray(crystal, dtype=np.int64)[step] + np.tile([0, 1], a.size)
    count = count.ravel()
    t = hit[step].astype(np.float32)
    keep = count > 0
    step, sipm, count, t = step[keep], sipm[keep], count[keep], t[keep]
    if not cfg.travel:
        return ev[step], sipm, t, count
    pe = np.repeat(np.arange(step.size), count)
    depth = (get["crystal_length"][step].astype(np.float32) - np.asarray(z_local, dtype=np.float32)[step])[pe]
    dt = propagation_time(depth, rng.random(pe.size, dtype=np.float32), table.cdf, table.times, table.dz)
    return ev[step][pe], sipm[pe], (t[pe] + dt).astype(np.float32), np.ones(pe.size, dtype=np.int64)
