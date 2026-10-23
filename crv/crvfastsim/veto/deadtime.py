"""CRV deadtime: the fraction of selected signal tracks lost to the veto (plan E4), by two methods.

    S (full)        per track, its own event's coincidences: vetoed, or its event fails an event-level flash gate.
    F (factorised)  per beam frame f, c_f = P(frame f vetoes a signal track | track t0 ~ the selected-signal t0
                    distribution); F = mean over frames of c_f, error std(c_f) / sqrt(N frames). Frames are any set of
                    signal-free or decorrelated coincidence records (the selected events themselves, or fast-chain
                    beam-only frames), so the CRV response and reco only run on beam frames, never per signal event.

F neglects the signal's own CRV activity and any t0-pileup correlation; S - F measures both (V-E4). The coverage is
evaluated on a time grid (default 0.25 ns) with difference arrays, chunked over frames: no per-event loop.

    python -m crvfastsim.veto.deadtime SEL.parquet [...] (--card JSON | --cosmic-ana CUTS.yaml CUTSET)
        [--frames FRAMES.parquet ...] [--pmin 85 --pmax 200] --outdir DIR
"""
from __future__ import annotations

import argparse
import json
import os

import awkward as ak
import numpy as np

from .apply import coinc_mask, gate, vetoed
from .card import VetoCard
from .metrics import fraction


def frame_coverage(coincs, card: VetoCard, t_hist, edges, chunk=4096):
    """c_f for every frame: the t0-weighted fraction of the grid where frame f vetoes a track or fails the event gate.
    coincs [F, C]; t_hist: normalised weights of the selected track t0 on the grid cells edges[:-1]..edges[1:]."""
    nf = len(coincs)
    c = np.zeros(nf)
    fail = ~gate(coincs, card)
    c[fail] = 1.0
    m = coinc_mask(coincs, card)
    tc = coincs["time"][m]
    has = np.flatnonzero((ak.to_numpy(ak.num(tc)) > 0) & ~fail)
    h, g0, G = edges[1] - edges[0], edges[0], len(edges) - 1
    for a in range(0, len(has), chunk):
        idx = has[a:a + chunk]
        t = tc[idx]
        k = np.repeat(np.arange(len(idx)), ak.to_numpy(ak.num(t)))
        t = ak.to_numpy(ak.flatten(t))
        lo = np.clip(np.ceil((t + card.dt_lo - g0) / h).astype(np.int64), 0, G)   # cell centres inside (lo, hi)
        hi = np.clip(np.ceil((t + card.dt_hi - g0) / h).astype(np.int64), 0, G)
        d = np.zeros((len(idx), G + 1), np.int32)
        np.add.at(d, (k, lo), 1)
        np.add.at(d, (k, hi), -1)
        cov = np.cumsum(d[:, :G], axis=1) > 0
        c[idx] = cov @ t_hist
    return c


def factorised(t_sel, coincs_frames, card: VetoCard, step=0.25):
    """Method F: dict with frac, err, frames; t_sel: 1-d selected track times."""
    t_sel = np.asarray(t_sel, float)
    lo = np.floor(t_sel.min()) - step if len(t_sel) else 0.0
    hi = np.ceil(t_sel.max()) + step if len(t_sel) else 1.0
    edges = np.arange(lo, hi + step, step) - step / 2
    w = np.histogram(t_sel, edges)[0].astype(float)
    w /= max(w.sum(), 1.0)
    c = frame_coverage(coincs_frames, card, w, edges)
    n = len(c)
    return {"frac": float(c.mean()) if n else float("nan"), "err": float(c.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "frames": int(n), "frames_gate_fail": int((~gate(coincs_frames, card)).sum()), "grid_ns": step}


def full(t_tracks, coincs, card: VetoCard, sel=None):
    """Method S: per selected track, vetoed by its own event or event failing the gate. t_tracks [E, T]."""
    v = vetoed(t_tracks, coincs, card)
    g = ak.broadcast_arrays(ak.Array(gate(coincs, card))[:, None], t_tracks)[0]
    v, g = ak.to_numpy(ak.flatten(v)), ak.to_numpy(ak.flatten(g))
    sel = np.ones(len(v), bool) if sel is None else sel
    out = fraction((v | ~g)[sel])
    out["vetoed_passing_gate"] = int((v & g)[sel].sum())
    out["gate_fail"] = int((~g)[sel].sum())
    return out


def read_coincs(paths):
    a = ak.concatenate([ak.from_parquet(p) for p in paths])
    co = a["coincs"]
    cols = [c for c in ("time", "timeStart", "timeEnd", "PEs", "nLayers", "sectorType") if c in co.fields]
    return a, ak.zip({c: co[c] for c in cols})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="selection parquet(s) from crvfastsim.veto.cosmic_ana (signal + beam)")
    ap.add_argument("--card", help="VetoCard JSON")
    ap.add_argument("--cosmic-ana", nargs=2, metavar=("CUTS_YAML", "CUTSET"))
    ap.add_argument("--gate-mode", choices=("event", "coinc"), help="override the card's gate_mode")
    ap.add_argument("--frames", nargs="*", help="frame parquet(s) with a coincs column (default: the selected events)")
    ap.add_argument("--pmin", type=float, default=None)
    ap.add_argument("--pmax", type=float, default=None)
    ap.add_argument("--step", type=float, default=0.25, help="coverage grid (ns)")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    if bool(a.card) == bool(a.cosmic_ana):
        ap.error("give exactly one of --card and --cosmic-ana")
    card = VetoCard.from_dict(json.load(open(a.card))) if a.card else VetoCard.from_mu2e_cosmic_ana(*a.cosmic_ana)
    if a.gate_mode:
        card = VetoCard.from_dict({**card.__dict__, "gate_mode": a.gate_mode})
    ev, co = read_coincs(a.inputs)
    t, p = ev["t_mid"], ak.to_numpy(ak.flatten(ev["p_mid"]))
    sel = np.ones(len(p), bool)
    if a.pmin is not None:
        sel &= p > a.pmin
    if a.pmax is not None:
        sel &= p < a.pmax
    S = full(t, co, card, sel)
    frames = read_coincs(a.frames)[1] if a.frames else co
    F = factorised(ak.to_numpy(ak.flatten(t))[sel], frames, card, a.step)
    out = {"card": card.__dict__ | {"sha": card.sha()}, "inputs": a.inputs, "frames_from": a.frames or "selected events",
           "pmin": a.pmin, "pmax": a.pmax, "selected": int(sel.sum()), "S": S, "F": F,
           "S_minus_F": S["frac"] - F["frac"]}
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(out, open(os.path.join(a.outdir, "deadtime_stats.json"), "w"), indent=1, default=float)
    print(f"{card.name} gate={card.gate_mode} ({card.sha()}), {out['selected']} selected tracks:\n"
          f"  S full       {S['k']} / {S['n']} = {S['frac']:.5f} [{S['lo']:.5f}, {S['hi']:.5f}]  "
          f"(vetoed {S['vetoed_passing_gate']}, gate fail {S['gate_fail']})\n"
          f"  F factorised {F['frac']:.5f} +- {F['err']:.5f} over {F['frames']} frames ({F['frames_gate_fail']} fail the gate)\n"
          f"  S - F        {out['S_minus_F']:+.5f}")


if __name__ == "__main__":
    main()
