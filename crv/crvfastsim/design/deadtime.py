"""CRV-induced deadtime of a design: a factorised rate model (method F), not a simulation.

The fast chain has no beam-induced (neutron, gamma) CRV simulation. This model carries the stock deadtime of the
nominal design to other designs with two components per sector s, each calibrated at the nominal design:
  accidental  R_acc = n_roads x C(N, k) x k x (m lambda)^k x tau^(k-1), with lambda = phi_s kappa (V / V_0)
              (Y / Y_0)^beta the singles rate of one counter (V counter volume, Y expected PE per MeV over the PE
              threshold relative to stock, beta the threshold exponent of the deposit spectrum), m = 2 counters per road (same plus adjacent), tau the
              sector's coincidence window, n_roads the counters per layer;
  correlated  R_cor = psi_s kappa (A / A_0) prod over absorber gaps exp(-a / lambda_att), multi-layer showers that
              cross the stack, attenuated by the absorbers (1 MeV photon attenuation lengths).
Deadtime (uniform signal t0 over the live window) = 1 - exp(-dt_veto x sum_s (R_acc + R_cor)).
Calibration: per-sector stock deadtime fractions D_s and the accidental share f_acc fix phi_s and psi_s; kappa is the
beam-intensity scenario. Without a calibration file only the rate ratio to the nominal design is defined
(uniform per sector area): ``rate_ratio``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

ATTEN_MM = {"G4_Al": 61.0, "MildSteel": 21.0, "G4_Pb": 13.0, "G4_POLYETHYLENE": 150.0}   # 1 MeV photons (XCOM)


@dataclass(frozen=True)
class DeadtimeConfig:
    kappa: float = 1.0          # beam intensity relative to the calibration
    f_acc: float = 0.5          # accidental share of the nominal deadtime
    beta: float = 1.0           # singles ~ (light yield)^beta
    roads: int = 2              # counters per layer in one road
    dt_veto: float = 150.0      # ns: veto window after a coincidence


def sector_terms(dg, sector_cuts, ly_per_mev, stock_required, thr_ratio=None):
    """Per sector: (n_roads, N, k, tau, counter volume [mm^3], sector area [mm^2], absorber transmission).
    k follows dg.design.coincidence_layers (veto.required_layers); thr_ratio [nsec]: design PE threshold over the
    stock threshold (a higher threshold acts like a lower light yield on the singles rate)."""
    from .veto import required_layers
    out = []
    for s, name in enumerate(dg.sector_names):
        b = dg.sector == s
        nl = int(dg.n_layers[s])
        k = int(required_layers(nl, stock_required[name], dg.design.coincidence_layers, dg.design.spare_layers))
        n_roads = int(b.sum() // nl)
        vol = float(np.prod(2 * dg.halflen[b][0]))
        area = float((4 * dg.halflen[b][:, 1] * dg.halflen[b][:, 2]).sum() / nl)
        ab = dg.abs_sector == s
        nmod = int(dg.module[b].max()) + 1
        gaps = 2 * dg.abs_halflen[ab][:, 0][: max(len(dg.abs_halflen[ab]) // max(nmod, 1), 0)]
        trans = float(np.exp(-gaps.sum() / ATTEN_MM.get(dg.absorber_material, 61.0)))
        out.append(dict(name=name, n_roads=n_roads, N=nl, k=k, tau=float(sector_cuts[name].max_dt), vol=vol,
                        area=area, trans=trans,
                        ly=float(ly_per_mev[s]) / (1.0 if thr_ratio is None else float(thr_ratio[s]))))
    return out


def _acc(t, lam, roads):
    return t["n_roads"] * math.comb(t["N"], t["k"]) * t["k"] * (roads * lam) ** t["k"] * t["tau"] ** (t["k"] - 1)


def evaluate(terms, terms0, cfg=DeadtimeConfig(), calibration=None):
    """dict: rate_ratio (always), and with a calibration deadtime, its per-sector rates and calibration source.
    terms / terms0: sector_terms of the design and of the nominal design. calibration: {"source": str,
    "deadtime": {sector: fraction} or "total": fraction (spread over sectors by area)}."""
    names = [t["name"] for t in terms0]
    a0 = np.array([t["area"] for t in terms0])
    src = None if calibration is None else calibration.get("source", "")
    if calibration is not None and "deadtime" in calibration:
        R = {n: -math.log(max(1.0 - float(calibration["deadtime"].get(n, 0.0)), 1e-300)) / cfg.dt_veto
             for n in names}
    else:   # a total spread by area (uncalibrated: any total, only the ratio is used)
        total = 0.01 if calibration is None else float(calibration["total"])
        rt = -math.log(max(1.0 - total, 1e-300)) / cfg.dt_veto
        R = {n: rt * a / a0.sum() for n, a in zip(names, a0)}
    r_nom, r_new, per = 0.0, 0.0, {}
    for t, t0 in zip(terms, terms0):
        R0 = R[t0["name"]]
        acc0, cor0 = cfg.f_acc * R0, (1 - cfg.f_acc) * R0
        lam0 = (acc0 / max(_acc(t0, 1.0, cfg.roads), 1e-300)) ** (1.0 / t0["k"]) if acc0 > 0 else 0.0
        lam = lam0 * cfg.kappa * (t["vol"] / t0["vol"]) * (t["ly"] / t0["ly"]) ** cfg.beta
        acc = _acc(t, lam, cfg.roads)
        cor = cor0 * cfg.kappa * (t["area"] / t0["area"]) * (t["trans"] / t0["trans"] if t0["trans"] > 0 else 1.0)
        per[t["name"]] = acc + cor
        r_nom += R0
        r_new += acc + cor
    out = dict(rate_ratio=r_new / r_nom if r_nom > 0 else float("nan"))
    if calibration is not None:
        out.update(deadtime=1.0 - math.exp(-cfg.dt_veto * r_new), calibration_source=src,
                   deadtime_sector={n: 1.0 - math.exp(-cfg.dt_veto * r) for n, r in per.items()})
    return out


def load_calibration(path):
    with open(path) as f:
        cal = json.load(f)
    if "deadtime" not in cal and "total" not in cal:
        raise ValueError(f"{path}: needs 'deadtime' {{sector: fraction}} or 'total'")
    return cal
