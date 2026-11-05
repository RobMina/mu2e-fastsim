"""Full fast calorimeter chain on saved CaloShowerSteps: readouts -> digis -> template fit -> hits -> clusters."""
from __future__ import annotations

import time

import numpy as np

from .cluster import ClusterConfig, Geometry, make_clusters
from .digi import DigiConfig, digitize, noise_pedestal
from .hits import make_hits
from .readout import ReadoutConfig, make_readouts
from .templatefit import FitConfig, fit_digis


def run_chain(steps, timing, adc_per_pe, pulse, noise_wf, geom: Geometry, rng, *, ro_cfg=ReadoutConfig(),
              table=None, add_noise=True, adc2mev=None, max_chi2=2.0, time4merge=6.0, fit_cfg=FitConfig(),
              cluster_cfg=ClusterConfig(), backend="jax"):
    """Run every stage for all events. ``adc_per_pe(sipm)`` gives ADCPerMeV/pePerMeV, ``adc2mev(sipm)`` the reco
    calibration (default 0.0625 for every SiPM); ``rng`` a NumPy Generator.

    Returns a dict with the flat ``digis``, ``reco`` (reco digis), ``hits`` and ``clusters`` tables.
    """
    n_events = timing["pbtime"].size
    clock = [time.perf_counter()]
    seconds = {}

    def lap(name):
        clock.append(time.perf_counter())
        seconds[name] = clock[-1] - clock[-2]
    ev, sipm, t, count = make_readouts(steps["event"], steps["crystal"], steps["energy_g4"], steps["energy_birks"],
                                       steps["time"], steps["z_local"], steps["lru"], steps["yields"], timing,
                                       ro_cfg, table=table, rng=rng)
    lap("readout")
    key = ev * 10000 + sipm
    ukey, row = np.unique(key, return_inverse=True)
    noise = None
    if add_noise:
        def offsets(length):
            return np.trunc(rng.random(len(length)) * (noise_wf.size - np.asarray(length))).astype(np.int64)
        noise = (noise_wf, noise_pedestal(noise_wf), offsets)
    d = digitize(row, t, timing["pbtime"][ev], adc_per_pe(sipm), ukey.size, pulse, DigiConfig(), weight=count,
                 noise=noise, backend=backend)
    d["event"], d["sipm"] = ukey[d["row"]] // 10000, ukey[d["row"]] % 10000
    lap("digitize")
    length = np.diff(d["offsets"])
    width = int(length.max(initial=1))
    padded = np.zeros((length.size, width), np.int64)
    member = np.repeat(np.arange(length.size), length)
    padded[member, np.arange(d["samples"].size) - d["offsets"][:-1][member]] = d["samples"]

    fit = fit_digis(d["t0"], padded, length, pulse, fit_cfg)
    lap("template_fit")
    chi_ok = (fit["chi2"] / np.maximum(fit["ndf"], 1).astype(np.float32)) <= max_chi2
    di, slot = np.nonzero(fit["valid"] & chi_ok[:, None])
    order = np.lexsort((fit["peak_time"][di, slot], di))
    di, slot = di[order], slot[order]
    f32 = np.float32
    scale = 0.0625 if adc2mev is None else adc2mev(d["sipm"][di])
    pbt = timing["pbtime"][d["event"][di]].astype(np.float64)
    reco = {"digi": di, "event": d["event"][di], "sipm": d["sipm"][di],
            "e": (fit["amp"][di, slot] * scale).astype(f32), "e_err": (fit["amp_err"][di, slot] * scale).astype(f32),
            "t": (fit["time"][di, slot] - pbt).astype(f32), "t_err": fit["time_err"][di, slot].astype(f32)}

    finite = np.isfinite(reco["e"]) & np.isfinite(reco["e_err"]) & np.isfinite(reco["t"]) & np.isfinite(reco["t_err"])
    finite &= (reco["e_err"] > 0) & (reco["t_err"] > 0)
    n_nonfinite = int((~finite).sum())
    reco = {k: v[finite] for k, v in reco.items()}
    h = make_hits(reco["event"].astype(np.int64), reco["sipm"].astype(np.int64), reco["t"], reco["t_err"], reco["e"],
                  reco["e_err"], n_events=n_events, n_crystals=geom.disk.size, time4merge=time4merge)
    hits = {"event": h["event_index"], "crystal": h["crystal"], "nsipm": h["nSiPMs"], "e": h["energy"],
            "e_err": h["energy_error"], "t": h["time"], "t_err": h["time_error"]}
    lap("hits")
    clusters = make_clusters({"event": hits["event"], "crystal": hits["crystal"], "time": hits["t"],
                              "energy": hits["e"]}, n_events, geom, cluster_cfg)
    lap("clusters")
    return {"seconds": seconds, "digis": d, "reco": reco, "hits": hits, "clusters": clusters,
            "n_refit_skipped": int(fit["refit_skipped"].sum()), "n_nonfinite_reco": n_nonfinite,
            "nonfinite_examples": {k: v[~finite][:5].tolist() for k, v in
                                   {"e": fit["amp"][di, slot] * scale, "t_err": fit["time_err"][di, slot],
                                    "amp_err": fit["amp_err"][di, slot]}.items()}}
