"""Figures of merit of one CRV design point: veto inefficiency, deadtime, light-yield margin and cost proxy.

  inefficiency       mean over the muon sample of P_miss (veto.py), with its standard error, leaving out the muons
                     that stop in the CRV before the CE-like region (stack_range; stopped_fraction); split into
                     ``inefficiency_covered`` (muons that cross at least one counter) and ``uncovered_fraction``;
                     per sector: each muon assigned to the highest sector (largest y of its crossings);
  deadtime           deadtime.py (absolute with a calibration file, ``deadtime_rate_ratio`` always);
  ly_margin          min over sectors of the design's minimum light yield along the counter (better end, normal
                     incidence, aged by ``aging_c1``) over the nominal design's (aged by ``aging_ref``);
  cost               mean of the four cost terms (module.cost_terms), each normalised to the nominal design.
The nominal design is the Design() of the same counters file, run through the same code.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np

from mu2efastsim import eloss, materials
from .. import runconfig as RC
from ..response import lookup
from . import deadtime as DT
from . import module as M
from . import optics as O
from . import veto as V

# CE-like proxy region (Mu2e mm): a cylinder about the DS axis from the stopping target to the tracker end
PROXY = dict(x=-3904.0, y=0.0, r=800.0, z0=5400.0, z1=11800.0)


@dataclass(frozen=True)
class FomConfig:
    optics: O.OpticsConfig = field(default_factory=O.OpticsConfig)
    deadtime: DT.DeadtimeConfig = field(default_factory=DT.DeadtimeConfig)
    aging_o1: float = 1.0      # light-yield factor for the inefficiency
    aging_c1: float = 0.94 ** 5   # Mu2e-II end of run (6 %/yr, 5 yr)
    aging_ref: float = 0.86    # nominal Run 2 counter at the end of Mu2e
    finder_cuts: bool = True   # the finder's slope and minimum-cluster-PE cuts in P_coinc (veto.finder_factor)
    stack_range: bool = True   # muons lose energy in the designed stack (veto.stack_range); those that stop before
                               # the CE-like region are left out of the inefficiency (they cannot reach the tracker)


def proxy_entry_t(origin, direction, proxy=PROXY):
    """Path parameter where each line enters the proxy cylinder (inf for lines that miss it)."""
    o = np.asarray(origin, np.float64) - [proxy["x"], proxy["y"], 0.0]
    d = np.asarray(direction, np.float64)
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1])
    c = o[:, 0] ** 2 + o[:, 1] ** 2 - proxy["r"] ** 2
    disc = b * b - 4 * a * c
    t = np.where((disc > 0) & (a > 0), (-b - np.sqrt(np.maximum(disc, 0.0))) / np.where(a > 0, 2 * a, 1.0), np.inf)
    return np.where(proxy_hits(origin, direction, proxy), np.maximum(t, 0.0), np.inf)


def proxy_hits(origin, direction, proxy=PROXY):
    """Lines (origin, direction) that pass through the proxy cylinder (axis along z)."""
    o = np.asarray(origin, np.float64) - [proxy["x"], proxy["y"], 0.0]
    d = np.asarray(direction, np.float64)
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1])
    c = o[:, 0] ** 2 + o[:, 1] ** 2 - proxy["r"] ** 2
    disc = b * b - 4 * a * c
    ok = (disc > 0) & (a > 0)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t1, t2 = (-b - sq) / np.where(ok, 2 * a, 1), (-b + sq) / np.where(ok, 2 * a, 1)
    z1, z2 = o[:, 2] + t1 * d[:, 2], o[:, 2] + t2 * d[:, 2]
    zlo, zhi = np.minimum(z1, z2), np.maximum(z1, z2)
    return ok & (zhi > proxy["z0"]) & (zlo < proxy["z1"])


def muons_from_gen(gen, proxy=PROXY):
    """CE-like proxy muons from a CRY gen table (pandas: pdg, x, y, z, px, py, pz): dict origin, direction, p."""
    mu = gen[np.abs(gen.pdg.to_numpy()) == 13]
    o = mu[["x", "y", "z"]].to_numpy(np.float64)
    pv = mu[["px", "py", "pz"]].to_numpy(np.float64)
    p = np.linalg.norm(pv, axis=1)
    d = pv / np.where(p > 0, p, 1.0)[:, None]
    k = proxy_hits(o, d, proxy) & (p > 0)
    return dict(origin=o[k], direction=d[k], p=p[k], n_generated_muons=int(len(mu)))


@dataclass
class Context:
    """What one evaluation reads: counters file, per-sector stock response and cuts, lookup tables."""
    counters: str = M.RUN2_COUNTERS
    sectors: dict = field(default_factory=lambda: dict(RC.RUN2_SECTORS))
    cuts: dict = field(default_factory=lambda: dict(RC.RUN2_CUTS))
    lookup_dir: str | None = None
    tables: dict = field(default_factory=dict)
    substitutions: dict = field(default_factory=dict)

    def table_for(self, name):
        """The sector's stock table if present, else the available table of the same reflector kind closest in
        length (recorded in substitutions)."""
        want = self.sectors[name].table
        if want not in self.tables:
            try:
                self.tables[want] = O.TableModel.from_lookup(lookup.load(want, self.lookup_dir))
            except FileNotFoundError:
                self.tables[want] = None
        if self.tables[want] is not None:
            return self.tables[want]
        import glob
        import os
        avail = [os.path.basename(p)[len("LookupTable_"):] for p in
                 glob.glob(os.path.join(self.lookup_dir or lookup.lookup_dir(), "LookupTable_*"))]
        refl = want.endswith("_1")
        cands = [a for a in avail if a.endswith("_1") == refl] or avail
        if not cands:
            raise FileNotFoundError(f"no lookup tables in {self.lookup_dir or lookup.lookup_dir()}")
        best = min(cands, key=lambda a: abs(float(a.split("_")[0]) - float(want.split("_")[0])))
        self.substitutions[want] = best
        if best not in self.tables:
            self.tables[best] = O.TableModel.from_lookup(lookup.load(best, self.lookup_dir))
        return self.tables[best]


def _stock_required(cuts):
    return {n: int(c.coincidence_layers) for n, c in cuts.items()}


def finder_config(dg, cuts, bars):
    """veto.p_miss `finder` argument: the sector cuts' slope window, slope difference and minimum cluster PE."""
    names = dg.sector_names
    return dict(dg=dg, bar=bars, min_slope=np.array([cuts[n].min_slope for n in names], float),
                max_slope=np.array([cuts[n].max_slope for n in names], float),
                max_slope_diff=np.array([cuts[n].max_slope_diff for n in names], float),
                min_pe=np.array([cuts[n].min_cluster_pes for n in names], float))


def sector_pe(dg, ctx, idx, pos, p, cfg, dtype):
    """Expected PE [m, 2] for the crossing rows (idx [m, 2] track, bar; pos [m, 6] local in/out)."""
    pe = np.zeros((len(idx), 2), dtype)
    sec = dg.sector[idx[:, 1]]
    for s, name in enumerate(dg.sector_names):
        r = np.flatnonzero(sec == s)
        if not len(r):
            continue
        sc = ctx.sectors[name]
        tab = ctx.table_for(name)
        pe[r] = O.expected_pe(pos[r, :3], pos[r, 3:], dg.halflen[idx[r, 1]], p[idx[r, 0]], tab, sc.yield_per_mev,
                              dg.fiber_diameter[s], dg.stock_fiber[s], M.readout_sides(sc.reflector), cfg, dtype)
    return pe


def light_yield(dg, ctx, cfg):
    """Per sector: (minimum better-end PE along the longest counter, PE per MeV at the counter centre)."""
    out_min, out_mev = [], []
    dedx = float(eloss.dedx_restricted(materials.POLYSTYRENE, np.array([1000.0]), np.inf)[0])
    for s, name in enumerate(dg.sector_names):
        b = np.flatnonzero(dg.sector == s)
        h = dg.halflen[b[np.argmax(dg.halflen[b, 2])]]
        sc = ctx.sectors[name]
        prof = O.light_yield_profile(h, ctx.table_for(name), sc.yield_per_mev, dg.fiber_diameter[s],
                                     dg.stock_fiber[s], M.readout_sides(sc.reflector), cfg)
        out_min.append(float(prof.min()))
        out_mev.append(float(prof[len(prof) // 2] / (dedx * 2 * h[0])))
    return np.array(out_min), np.array(out_mev)


def evaluate(design, muons, ctx=None, cfg=FomConfig(), calibration=None, dtype=np.float64, nominal=None):
    """Metrics of one design point (dict). muons: dict origin [n, 3], direction [n, 3], p [n] (muons_from_gen)."""
    t0 = time.perf_counter()
    ctx = ctx or Context()
    yields = {n: s.yield_per_mev for n, s in ctx.sectors.items()}
    dg = M.build(design, ctx.counters, yields)
    dg0 = nominal or M.build(M.Design(), ctx.counters, yields)
    o1 = dataclasses.replace(cfg.optics, aging=cfg.optics.aging * cfg.aging_o1)
    idx, pos = V.sector_crossings(dg, muons["origin"], muons["direction"], dtype)
    n = len(muons["p"])
    stopped = np.zeros(n, bool)
    if cfg.stack_range:
        keep, stopped = V.stack_range(dg, muons["origin"], muons["direction"], muons["p"], idx, pos,
                                      proxy_entry_t(muons["origin"], muons["direction"]))
        idx, pos = idx[keep], pos[keep]
    pe = sector_pe(dg, ctx, idx, pos, np.asarray(muons["p"], np.float64), o1, dtype)
    names = dg.sector_names
    req = _stock_required(ctx.cuts)
    thr0 = np.array([ctx.cuts[nm].pe_threshold for nm in names], np.float64)
    thr = np.full_like(thr0, design.pe_threshold) if design.pe_threshold is not None else thr0
    k = np.array([V.required_layers(int(dg.n_layers[s]), req[nm], design.coincidence_layers, design.spare_layers)
                  for s, nm in enumerate(names)])
    pm, (gt, gs, pc) = V.p_miss(n, idx[:, 0], dg.sector[idx[:, 1]], dg.layer[idx[:, 1]], pe, dg.n_layers, thr, k,
                                finder_config(dg, ctx.cuts, idx[:, 1]) if cfg.finder_cuts else None)
    covered = np.zeros(n, bool)
    covered[idx[:, 0]] = True
    ymax = np.full(n, -np.inf)
    cy = dg.center[idx[:, 1], 1]
    np.maximum.at(ymax, idx[:, 0], cy)
    top = np.full(n, -1)
    if len(idx):
        order = np.lexsort((cy, idx[:, 0]))
        last = np.r_[idx[order[1:], 0] != idx[order[:-1], 0], True]
        top[idx[order[last], 0]] = dg.sector[idx[order[last], 1]]
    live = ~stopped                                    # muons that reach the CE-like region
    nl_ = max(int(live.sum()), 1)
    per = {nm: float(pm[(top == s) & live].sum() / nl_) for s, nm in enumerate(names)}
    se = float(pm[live].std(ddof=1) / np.sqrt(nl_)) if nl_ > 1 else float("nan")
    # light yield (C1) and deadtime
    c1 = dataclasses.replace(cfg.optics, aging=cfg.optics.aging * cfg.aging_c1)
    c0 = dataclasses.replace(cfg.optics, aging=cfg.optics.aging * cfg.aging_ref)
    ly, _ = light_yield(dg, ctx, c1)
    ly0, _ = light_yield(dg0, ctx, c0)
    _, ly_mev = light_yield(dg, ctx, cfg.optics)      # deadtime: both designs at the same aging
    _, ly0_mev = light_yield(dg0, ctx, cfg.optics)
    terms = DT.sector_terms(dg, ctx.cuts, ly_mev, req, thr / thr0)
    terms0 = DT.sector_terms(dg0, ctx.cuts, ly0_mev, req)
    dt = DT.evaluate(terms, terms0, cfg.deadtime, calibration)
    # cost
    refl = {nm: ctx.sectors[nm].reflector for nm in names}
    dens = materials.get(dg.absorber_material).rho
    ct, ct0 = M.cost_terms(dg, refl, dens), M.cost_terms(dg0, refl, materials.get(dg0.absorber_material).rho)
    rel = {k_: (ct[k_] / ct0[k_] if ct0[k_] else 1.0) for k_ in ct}
    out = dict(
        inefficiency=float(pm[live].mean()) if live.any() else float("nan"), inefficiency_err=se,
        inefficiency_covered=float(pm[covered & live].sum() / nl_), uncovered_fraction=float((~covered[live]).mean()),
        inefficiency_sector=per, n_muons=n, n_crossings=int(len(idx)), stopped_fraction=float(stopped.mean()),
        deadtime_rate_ratio=dt["rate_ratio"], ly_min=dict(zip(names, ly.tolist())),
        ly_margin=float(np.min(ly / ly0)), cost=float(np.mean(list(rel.values()))), cost_terms=ct, cost_rel=rel,
        n_bars=int(dg.nbars), module_pitch=dict(zip(names, dg.module_pitch.tolist())),
        coincidence=dict(zip(names, [f"{int(kk)} of {int(nn)}, {tt:g} PE" for kk, nn, tt in zip(k, dg.n_layers, thr)])),
        table_substitutions=dict(ctx.substitutions), dtype=np.dtype(dtype).name, wall_s=time.perf_counter() - t0)
    for k_ in ("deadtime", "deadtime_sector", "calibration_source"):
        if k_ in dt:
            out[k_] = dt[k_]
    return out
