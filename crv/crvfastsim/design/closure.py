"""Closure of the design optics model against the full response chain on the same chords.

For an untilted design whose counters match stock lookup tables (same length and reflector), every muon crossing
becomes one CrvStep (visible energy: Bethe mean, or a sampled Geant4-like loss) and goes through the ported stock
photon and SiPM stages (response.photons, response.sipm). Per (muon, bar, end) the full-chain PE is compared with the
model's expected PE; per (muon, sector, layer) the fraction of layers above the sector threshold with the model's
Poisson probability; per (muon, sector) the coincidence fraction with P_coinc. Fits pe_scale (sum full / sum model).
Sectors whose stock table is absent are skipped (the full chain cannot read a counter longer than its table).
Each muon gets its own time slot (MUON_SPACING_NS apart), so SiPM pixel recovery never couples two muons: the
full chain sees each muon as its own event, as the per-muon model does.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from scipy.special import pdtr

from mu2efastsim import eloss, materials
from ..response import lookup
from ..response import photons as PH
from ..response import sipm as SI
from . import fom as F
from . import module as M
from . import veto as V

MUON_SPACING_NS = 1000.0     # >> SiPM recovery (13.3 ns) and trap lifetimes (50 ns)
T0_NS = 500.0


def _steps(dg, idx, pos, p, edep):
    c = dg.center[idx[:, 1]]
    A = dg.axes[dg.sector[idx[:, 1]]]                        # [m, 3, 3]
    x0 = c + np.einsum("mi,mij->mj", pos[:, :3], A)
    x1 = c + np.einsum("mi,mij->mj", pos[:, 3:], A)
    d = (x1 - x0) / np.linalg.norm(x1 - x0, axis=1)[:, None]
    path = np.linalg.norm(x1 - x0, axis=1)
    t0 = T0_NS + MUON_SPACING_NS * idx[:, 0].astype(np.float64)
    st = pd.DataFrame({"bar": idx[:, 1].astype(np.int64), "edep": edep, "t0": t0, "len": path})
    st["t1"] = st.t0 + path / 299.79
    for j, a in enumerate("xyz"):
        st[f"{a}0"], st[f"{a}1"] = x0[:, j], x1[:, j]
        st[f"p{a}0"] = d[:, j] * p[idx[:, 0]]
    st["p1"] = p[idx[:, 0]]
    return st


def run(design, muons, ctx, cfg=F.FomConfig(), photon_map=None, mode="mean", seed=1, dtype=np.float64):
    """dict of closure statistics (per sector and overall). mode: 'mean' (Bethe mean visible energy) or 'sampled'."""
    if design.tilt_deg:
        raise ValueError("closure needs an untilted design")
    yields = {n: s.yield_per_mev for n, s in ctx.sectors.items()}
    dg = M.build(design, ctx.counters, yields)
    geo = M.to_crv_geometry(dg)
    names = dg.sector_names
    for n in names:
        ctx.table_for(n)
    ok = [s for s, n in enumerate(names) if ctx.sectors[n].table not in ctx.substitutions
          and np.allclose(dg.halflen[dg.sector == s, 2], ctx.table_for(n).half[2])]
    rng = np.random.default_rng(seed)
    p = np.asarray(muons["p"], np.float64)
    idx, pos = V.sector_crossings(dg, muons["origin"], muons["direction"], np.float64)
    keep = np.isin(dg.sector[idx[:, 1]], ok)
    idx, pos = idx[keep], pos[keep]
    path = np.linalg.norm(pos[:, 3:] - pos[:, :3], axis=1)
    if mode == "mean":
        edep = eloss.dedx_restricted(materials.POLYSTYRENE, p[idx[:, 0]], np.inf) * path
    else:
        edep = eloss.sample_loss(rng, materials.POLYSTYRENE, p[idx[:, 0]], path, 1e9)
    steps = _steps(dg, idx, pos, p, edep)
    if not ok:
        raise ValueError("no sector has its stock lookup table: nothing to close")
    tables = {ctx.sectors[names[s]].table: lookup.load(ctx.sectors[names[s]].table, ctx.lookup_dir) for s in ok}
    first = next(iter(tables))
    sectors = {n: (ctx.sectors[n] if i in ok else             # sectors without steps: any loaded table
                   dataclasses.replace(ctx.sectors[n], table=first, reflector=tables[first].reflector))
               for i, n in enumerate(names)}
    rcfg = dataclasses.replace(PH.ResponseConfig(), sectors=sectors, photon_yield_scale=cfg.optics.photon_yield_scale)
    step_i, chan, t = PH.make_photons(steps, np.full(len(steps), 13), geo, rcfg, tables, np.zeros(4 * geo.nbars), rng)
    t_end = float(steps.t0.max()) + MUON_SPACING_NS if len(steps) else 1e5
    ch = SI.simulate(chan, t, step_i, SI.photon_map(photon_map), SI.SiPMConfig(), rng, t_end=t_end)
    full = np.zeros((len(steps), 2))
    good = ch["photon"] >= 0
    side = np.where(np.isin(ch["chan"][good] % 4, (0, 2)), 0, 1)
    np.add.at(full, (ch["photon"][good].astype(np.int64), side), ch["pe"][good])
    model = F.sector_pe(dg, ctx, idx, pos.astype(dtype), p, cfg.optics, dtype).astype(np.float64)
    out = dict(mode=mode, n_crossings=int(len(idx)), sectors={}, dtype=np.dtype(dtype).name)
    req = F._stock_required(ctx.cuts)
    sec = dg.sector[idx[:, 1]]
    for s in ok:
        r = sec == s
        if not r.any():
            continue
        nm = names[s]
        fsum, msum = full[r].sum(), model[r].sum()
        ratio = fsum / msum if msum > 0 else float("nan")
        per = full[r].sum(1) / np.maximum(model[r].sum(1), 1e-9)
        thr = float(ctx.cuts[nm].pe_threshold)
        # layer level: (muon, layer) sums per end
        key = idx[r, 0] * 16 + dg.layer[idx[r, 1]]
        uk, ki = np.unique(key, return_inverse=True)
        fl, ml = np.zeros((len(uk), 2)), np.zeros((len(uk), 2))
        np.add.at(fl, ki, full[r])
        np.add.at(ml, ki, model[r] * (ratio if mode == "mean" else 1.0))
        hit_full = (fl >= thr).any(1)
        p_model = 1 - np.where(ml > 0, pdtr(np.ceil(thr) - 1, ml), 1.0).prod(1)
        out["sectors"][nm] = dict(
            table=ctx.sectors[nm].table, crossings=int(r.sum()), pe_ratio=float(ratio),
            pe_ratio_err=float(per.std(ddof=1) / np.sqrt(r.sum())) if r.sum() > 1 else float("nan"),
            pe_full_mean=float(full[r].sum(1).mean()), pe_model_mean=float(model[r].sum(1).mean()),
            layers=int(len(uk)), layer_hit_full=float(hit_full.mean()), layer_hit_model=float(p_model.mean()),
            layer_hit_err=float(np.sqrt(max(hit_full.mean() * (1 - hit_full.mean()), 1e-12) / len(uk))),
            required_layers=req[nm])
    tot_f = sum(v["pe_full_mean"] * v["crossings"] for v in out["sectors"].values())
    tot_m = sum(v["pe_model_mean"] * v["crossings"] for v in out["sectors"].values())
    out["pe_scale_fit"] = float(tot_f / tot_m) if tot_m else float("nan")
    out["skipped_sectors"] = [n for s, n in enumerate(names) if s not in ok]
    return out
