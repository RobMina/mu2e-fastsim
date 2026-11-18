"""The design model against a stock Geant4 + CRV response and reco sample of the same generated muons (paired).

Stock input: TFileService files holding the CrvStepTableDumper trees (``dump/gen``) and the CrvRecoTableDumper trees
(``crvdump/pulses``, ``crvdump/clusters``, ``crvdump/events``) of one job, every generated event kept. The geometry of
the stock job must be the design's (the nominal Design reproduces the stock bar indices, so bar b is the same counter
in both arms).

Fast side: every generated muon (|pdg| 13) of an event as a straight line from its generated state; expected PE per
counter end (optics), per (event, sector, layer, end) summed over the crossed counters, P(layer hit) and P(k of N).
Per sector, paired over events:
  pe            stock PE (pulses on the sector's counters) over the model's, per layer and read-out end, as the ratio
                of sums with its paired standard error (events where the model crosses that layer);
  layer_hit     fraction of (event, layer) rows with an end at or above the PE threshold, stock vs model mean P;
  coincidence   fraction of events with a stock coincidence cluster in the sector (a member pulse on one of its
                counters) vs the model's mean P_coinc, over events whose muons cross the sector.
Electrons, photons and secondaries reach the stock arm only; the model has primary muons (W9 primary_muon_lines).
"""
from __future__ import annotations

import awkward as ak
import numpy as np
from scipy.special import pdtr

from .. import tables as T
from ..reco import reftables as RT
from . import fom as F
from . import module as M
from . import veto as V


def _ekey(run, subrun, event):
    return (np.asarray(run, np.int64) << 42) | (np.asarray(subrun, np.int64) << 21) | np.asarray(event, np.int64)


def read_stock(paths):
    """dict: events (keys), muons (event index, origin, direction, p), pulses (event index, bar, side, pe), clusters
    (event index, member bars as a list per cluster)."""
    ev = RT.read(paths, "events", ["run", "subrun", "event"])
    keys = np.unique(_ekey(ak.to_numpy(ev.run), ak.to_numpy(ev.subrun), ak.to_numpy(ev.event)))
    gen = T.read_many(paths, "gen", T.KEY + ["pdg", "x", "y", "z", "px", "py", "pz"])
    gen = gen[np.abs(gen.pdg.to_numpy()) == 13]
    gi = np.searchsorted(keys, _ekey(gen.run, gen.subrun, gen.event))
    pv = gen[["px", "py", "pz"]].to_numpy(np.float64)
    p = np.linalg.norm(pv, axis=1)
    muons = dict(event=gi, origin=gen[["x", "y", "z"]].to_numpy(np.float64), direction=pv / p[:, None], p=p)
    pu = RT.read(paths, "pulses", ["run", "subrun", "event", "idx", "bar", "sipm", "pe"])
    pe_ev = np.searchsorted(keys, _ekey(ak.to_numpy(pu.run), ak.to_numpy(pu.subrun), ak.to_numpy(pu.event)))
    pulses = dict(event=pe_ev, idx=ak.to_numpy(pu.idx).astype(np.int64), bar=ak.to_numpy(pu.bar).astype(np.int64),
                  side=np.where(np.isin(ak.to_numpy(pu.sipm) % 4, (0, 2)), 0, 1), pe=ak.to_numpy(pu.pe).astype(float))
    cl = RT.read(paths, "clusters", ["run", "subrun", "event", "pulses"])
    cl_ev = np.searchsorted(keys, _ekey(ak.to_numpy(cl.run), ak.to_numpy(cl.subrun), ak.to_numpy(cl.event)))
    # member pulse keys -> bars, through (event, pulse idx)
    pk = pulses["event"] * (1 << 20) + pulses["idx"]
    order = np.argsort(pk)
    n = ak.to_numpy(ak.num(cl.pulses))
    mem = ak.to_numpy(ak.flatten(cl.pulses)).astype(np.int64)
    mk = np.repeat(cl_ev, n) * (1 << 20) + mem
    pos = order[np.searchsorted(pk[order], mk)]
    if len(pos) and not np.array_equal(pk[pos], mk):
        raise ValueError("a cluster member pulse is missing from crvdump/pulses")
    clusters = dict(event=np.repeat(cl_ev, n), cluster=np.repeat(np.arange(len(n)), n), bar=pulses["bar"][pos])
    return dict(n_events=len(keys), muons=muons, pulses=pulses, clusters=clusters, paths=list(paths))


def _ratio(a, b):
    """Ratio of sums of paired samples a, b with its delta-method standard error."""
    sa, sb, n = float(a.sum()), float(b.sum()), len(a)
    if n < 2 or sb <= 0:
        return float("nan"), float("nan")
    r = sa / sb
    return r, float(np.std(a - r * b, ddof=1) * np.sqrt(n) / sb)


def compare(stock, design=M.Design(), ctx=None, cfg=F.FomConfig(), dtype=np.float64, sectors=None):
    """Per-sector paired comparison (dict) of ``stock`` (read_stock) with the design model on the same muons."""
    ctx = ctx or F.Context()
    yields = {n: s.yield_per_mev for n, s in ctx.sectors.items()}
    dg = M.build(design, ctx.counters, yields)
    mu = stock["muons"]
    idx, pos = V.sector_crossings(dg, mu["origin"], mu["direction"], dtype)
    pe = F.sector_pe(dg, ctx, idx, pos, mu["p"], cfg.optics, dtype).astype(np.float64)
    names = dg.sector_names
    req = F._stock_required(ctx.cuts)
    nev, nl = stock["n_events"], int(dg.n_layers.max())
    ev_x = mu["event"][idx[:, 0]]
    sec_x, lay_x = dg.sector[idx[:, 1]], dg.layer[idx[:, 1]]
    pu, cl = stock["pulses"], stock["clusters"]
    sec_p, lay_p = dg.sector[pu["bar"]], dg.layer[pu["bar"]]
    out = dict(n_events=nev, n_muons=len(mu["p"]), n_crossings=int(len(idx)), dtype=np.dtype(dtype).name,
               pe_scale=cfg.optics.pe_scale, design=design.as_dict(), sectors={})
    for s, nm in enumerate(names):
        if sectors and nm not in sectors:
            continue
        xs = sec_x == s
        if not xs.any():
            continue
        thr = float(design.pe_threshold or ctx.cuts[nm].pe_threshold)
        k = int(V.required_layers(int(dg.n_layers[s]), req[nm], design.coincidence_layers))
        # (event, layer, end) sums, both arms, on the events the model crosses
        fast = np.zeros((nev, nl, 2))
        np.add.at(fast, (ev_x[xs], lay_x[xs]), pe[xs])
        st = np.zeros((nev, nl, 2))
        ps = sec_p == s
        np.add.at(st, (pu["event"][ps], lay_p[ps], pu["side"][ps]), pu["pe"][ps])
        crossed = np.zeros((nev, nl), bool)
        crossed[ev_x[xs], lay_x[xs]] = True
        rows = np.argwhere(crossed)
        per_layer = []
        for j in range(int(dg.n_layers[s])):
            r = rows[rows[:, 1] == j, 0]
            if len(r) < 2:
                continue
            ends = [_ratio(st[r, j, e], fast[r, j, e]) for e in (0, 1)]
            per_layer.append(dict(layer=j, rows=int(len(r)), pe_stock=float(st[r, j].sum(1).mean()),
                                  pe_model=float(fast[r, j].sum(1).mean()), ratio_end0=ends[0][0],
                                  ratio_end0_err=ends[0][1], ratio_end1=ends[1][0], ratio_end1_err=ends[1][1]))
        hit_st = (st[rows[:, 0], rows[:, 1]] >= thr).any(1)
        p_end = 1.0 - np.where(fast > 0, pdtr(np.ceil(thr) - 1.0, fast), 1.0)
        p_hit = 1.0 - (1.0 - p_end).prod(2)
        ph = p_hit[rows[:, 0], rows[:, 1]]
        evs = np.unique(ev_x[xs])
        pc = V.p_coincidence(fast[evs], np.full(len(evs), thr), np.full(len(evs), k))
        has_cl = np.zeros(nev, bool)
        has_cl[cl["event"][dg.sector[cl["bar"]] == s]] = True
        c_st = has_cl[evs]
        out["sectors"][nm] = dict(
            events=int(len(evs)), rule=f"{k} of {int(dg.n_layers[s])}, {thr:g} PE", pe=per_layer,
            pe_ratio=_ratio(st[rows[:, 0], rows[:, 1]].sum(1), fast[rows[:, 0], rows[:, 1]].sum(1)),
            layer_hit_stock=float(hit_st.mean()), layer_hit_model=float(ph.mean()),
            layer_hit_diff_err=float(np.std(hit_st - ph, ddof=1) / np.sqrt(len(ph))) if len(ph) > 1 else float("nan"),
            coinc_stock=float(c_st.mean()), coinc_model=float(pc.mean()),
            coinc_diff_err=float(np.std(c_st - pc, ddof=1) / np.sqrt(len(pc))) if len(pc) > 1 else float("nan"),
            miss_stock=int((~c_st).sum()), miss_model=float((1 - pc).sum()))
    return out


def geometry_check(paths, design=M.Design(), ctx=None, max_steps=500_000, tol=0.01):
    """Stock CrvStep midpoints (dump/steps) in the design bar of the same index, in its local frame: the fraction
    inside (within tol mm; the dump stores float32 positions) and the largest excess. 1.0 means the stock geometry
    is the design's."""
    ctx = ctx or F.Context()
    dg = M.build(design, ctx.counters, {n: s.yield_per_mev for n, s in ctx.sectors.items()})
    st = T.read_many(paths, "steps", ["bar", "x0", "y0", "z0", "x1", "y1", "z1"]).head(max_steps)
    b = st.bar.to_numpy().astype(np.int64)
    mid = 0.5 * (st[["x0", "y0", "z0"]].to_numpy(np.float64) + st[["x1", "y1", "z1"]].to_numpy(np.float64))
    loc = np.einsum("mj,mij->mi", mid - dg.center[b], dg.axes[dg.sector[b]])
    excess = (np.abs(loc) - dg.halflen[b]).max(1)
    return dict(steps=int(len(b)), inside=float((excess <= tol).mean()) if len(b) else float("nan"),
                max_excess_mm=float(excess.max()) if len(b) else float("nan"))


def inefficiency(stock, gen_paths, design=M.Design(), ctx=None, cfg=F.FomConfig(), dtype=np.float64,
                 proxy=F.PROXY):
    """CE-proxy veto inefficiency, paired over generated events: events with a proxy muon (fom.proxy_hits) among
    every generated event of the gen dump; stock miss = no coincidence cluster anywhere (events absent from the
    stock reco, which keeps only events with CRV steps, are misses); model = prod over the event's muons of P_miss."""
    ctx = ctx or F.Context()
    dg = M.build(design, ctx.counters, {n: s.yield_per_mev for n, s in ctx.sectors.items()})
    gen = T.read_many(gen_paths, "gen", T.KEY + ["pdg", "x", "y", "z", "px", "py", "pz"])
    ek_all = _ekey(gen.run, gen.subrun, gen.event)
    n_gen_events = int(len(np.unique(ek_all)))
    gen = gen[np.abs(gen.pdg.to_numpy()) == 13]
    o = gen[["x", "y", "z"]].to_numpy(np.float64)
    pv = gen[["px", "py", "pz"]].to_numpy(np.float64)
    p = np.linalg.norm(pv, axis=1)
    d = pv / np.where(p > 0, p, 1.0)[:, None]
    ek = _ekey(gen.run, gen.subrun, gen.event)
    sel = np.isin(ek, np.unique(ek[F.proxy_hits(o, d, proxy) & (p > 0)]))
    o, d, p, ek = o[sel], d[sel], p[sel], ek[sel]
    idx, pos = V.sector_crossings(dg, o, d, dtype)
    pe = F.sector_pe(dg, ctx, idx, pos, p, cfg.optics, dtype)
    names = dg.sector_names
    req = F._stock_required(ctx.cuts)
    thr0 = np.array([ctx.cuts[nm].pe_threshold for nm in names], np.float64)
    thr = np.full_like(thr0, design.pe_threshold) if design.pe_threshold is not None else thr0
    k = np.array([V.required_layers(int(dg.n_layers[s]), req[nm], design.coincidence_layers)
                  for s, nm in enumerate(names)])
    pm, _ = V.p_miss(len(p), idx[:, 0], dg.sector[idx[:, 1]], dg.layer[idx[:, 1]], pe, dg.n_layers, thr, k)
    ue, inv = np.unique(ek, return_inverse=True)
    pm_ev = np.ones(len(ue))
    np.multiply.at(pm_ev, inv, pm)
    # stock: events with a cluster (stock keys are run/subrun/event of the reco dump)
    ev = RT.read(stock["paths"], "events", ["run", "subrun", "event"])
    sk = np.unique(_ekey(ak.to_numpy(ev.run), ak.to_numpy(ev.subrun), ak.to_numpy(ev.event)))
    has = np.zeros(len(sk), bool)
    has[np.unique(stock["clusters"]["event"])] = True
    pos_ = np.searchsorted(sk, ue)
    found = (pos_ < len(sk)) & (sk[np.minimum(pos_, len(sk) - 1)] == ue)
    miss_st = ~(found & has[np.minimum(pos_, len(sk) - 1)])
    n = len(ue)
    diff = miss_st.astype(float) - pm_ev
    return dict(generated_events=n_gen_events, proxy_events=int(n), stock_misses=int(miss_st.sum()),
                stock_inefficiency=float(miss_st.mean()), model_expected_misses=float(pm_ev.sum()),
                model_inefficiency=float(pm_ev.mean()),
                diff=float(diff.mean()), diff_err=float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
                both_miss=int((miss_st & (pm_ev > 0.5)).sum()))
