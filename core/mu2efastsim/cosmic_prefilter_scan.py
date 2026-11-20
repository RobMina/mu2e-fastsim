"""Loss and Geant4 time saved by the cosmic pre-filter (cosmic_prefilter) as functions of its parameters.

Three steps, each restartable per shard:

    features  GEN [GEN ...] --gdml G --out F.npz [--dtype float32|float64] [--margins 0,100,...]
              per-primary features (cosmic_prefilter.Features) of the CRY primaries of every generated event: a flat
              `dump/gen` table (crv CrvStepTableDumper, e.g. a g4ref gen stream) or a PrintModule GenParticle listing
              (`.txt` / `.txt.gz`, see read_printgen)
    census    --crv CRVSTEPS --trk SGSTEPS --out C.npz
              the stock PrimaryFilter decision of every accepted event of a reference (dts dumped by the crv and
              tracker table dumpers), per term, with the CRY primaries whose descendants made it
    timing    --db timing.db --gen GEN.txt.gz --dts DTS.txt.gz --out T.npz [--skip-first 20]
              per-event Geant4 (g4run) time of a timed stock job, joined to its accepted events
    curve     --features F.npz ... --census C.npz ... [--timing-features TF.npz ... --timing T.npz ...] --out curve.json
              every parameter set of the grid: loss of accepted events (total, per term, and split into non-muon,
              muon near-miss and muon range-out components) on the reference, Geant4 time saved on the timed sample

The stock PrimaryFilter of the extracted chain (Production primary/prolog.fcl + ExtractedCRY.fcl): an event passes
when one SimParticle has >= 12 StrawGasSteps with ionising energy > 1 eV and 50 < |p| < 1e6 MeV/c (trk), or >= 1
CrvStep with visible energy > 0.1 MeV and the same momentum window (crv), or >= 45 MeV of calorimeter energy (calo,
not dumped: an accepted event with neither of the other terms is counted as calo).
"""
import argparse
import gzip
import itertools
import json
import re
import sqlite3
import time

import numpy as np

from . import cosmic_prefilter as cp

EVENT_STRIDE = 10_000_000          # event key = subrun * EVENT_STRIDE + event
DEFAULT_MARGINS = (0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
TRK = dict(min_edep=1e-6, min_p=50.0, max_p=1e6, min_steps=12)
CRV = dict(min_edep=0.1, min_p=50.0, max_p=1e6, min_steps=1)
MAX_PRIMARIES = 63                 # primaries per event representable in the accepting-primary bit masks
KEY_STRIDE = 1_000_000             # particle id = event key * KEY_STRIDE + SimParticle key


def event_key(subrun, event):
    return np.asarray(subrun, np.int64) * EVENT_STRIDE + np.asarray(event, np.int64)


def particle_id(ev, key):
    """ev * KEY_STRIDE + key as int64; raises if a key is >= KEY_STRIDE or the packed id would overflow int64."""
    ev, key = np.asarray(ev, np.int64), np.asarray(key, np.int64)
    if key.size and key.max() >= KEY_STRIDE:
        raise ValueError(f"SimParticle key {int(key.max())} >= {KEY_STRIDE}: particle ids would collide")
    if ev.size and np.abs(ev).max() > (np.iinfo(np.int64).max - KEY_STRIDE) // KEY_STRIDE:
        raise ValueError(f"event key {int(np.abs(ev).max())} too large: particle ids would overflow int64")
    return ev * KEY_STRIDE + key


# ----------------------------------------------------------------------------------------------- readers
def read_gen_table(path):
    """CRY primaries of a flat `dump/gen` table as a dict of columns (event = event key, idx, pdg, x..pz, e)."""
    import uproot
    a = uproot.open(path)["dump/gen"].arrays(["subrun", "event", "idx", "pdg", "x", "y", "z", "px", "py", "pz", "e"],
                                            library="np")
    out = {k: a[k] for k in ("idx", "pdg", "x", "y", "z", "px", "py", "pz", "e")}
    out["event"] = event_key(a["subrun"], a["event"])
    return out


_EVENT_LINE = re.compile(r"PrintModule Run/Subrun/Event\s+(\d+)\s+(\d+)\s+(\d+)")


def read_printgen(path):
    """CRY primaries from a PrintModule GenParticle listing (genParticlePrinter verbose 1): one line per particle,
    `index pdg x y z px py pz t tau generator`, positions and momenta rounded to 0.1 mm and 0.1 MeV/c; e from |p|
    and the particle mass. Also returns every event key seen (events with no particle line included)."""
    op = gzip.open if str(path).endswith(".gz") else open
    rows, events = [], []
    sub = ev = None
    with op(path, "rt") as fh:
        for line in fh:
            m = _EVENT_LINE.search(line)
            if m:
                sub, ev = int(m.group(2)), int(m.group(3))
                events.append(event_key(sub, ev))
                continue
            f = line.split()
            if ev is None or len(f) != 11 or not f[0].isdigit():
                continue
            try:
                vals = [float(x) for x in f[2:10]]
            except ValueError:
                continue
            rows.append((event_key(sub, ev), int(f[0]), int(f[1]), *vals[:6]))
    a = np.array(rows, dtype=[("event", np.int64), ("idx", np.int32), ("pdg", np.int32), ("x", float), ("y", float),
                              ("z", float), ("px", float), ("py", float), ("pz", float)])
    out = {k: a[k] for k in a.dtype.names}
    p = np.sqrt(out["px"] ** 2 + out["py"] ** 2 + out["pz"] ** 2)
    m = np.array([cp.MASS[abs(int(c))] for c in out["pdg"]], float)
    out["e"] = np.sqrt(p * p + m * m)
    return out, np.unique(np.asarray(events, np.int64))


# ----------------------------------------------------------------------------------------------- census
def _primary_of(ev, key, parent, genidx):
    """Gen index of the primary ancestor of every (event, key) particle (-1 when the chain breaks)."""
    gid = particle_id(ev, key)
    order = np.argsort(gid)
    gs = gid[order]
    n = len(gid)
    pidx = np.full(n, -1, np.int64)
    has = parent >= 0
    if n and has.any():
        pg = particle_id(ev[has], parent[has])
        pos = np.minimum(np.searchsorted(gs, pg), n - 1)
        pidx[has] = np.where(gs[pos] == pg, order[pos], -1)
    cur = np.arange(n)
    for _ in range(1000):
        nxt = pidx[cur]
        move = nxt >= 0
        if not move.any():
            break
        cur = np.where(move, nxt, cur)
    return genidx[cur]


def _accepting(ev, sp, n_ok, min_steps):
    """(event, sp) pairs with at least min_steps good steps."""
    key = particle_id(ev[n_ok], sp[n_ok])
    u, c = np.unique(key, return_counts=True)
    u = u[c >= min_steps]
    return u // KEY_STRIDE, u % KEY_STRIDE


def census(crv_path, trk_path):
    """Stock PrimaryFilter terms of every accepted event of one reference shard (see the module docstring).

    Returns a dict: event (keys of the accepted events), crv, trk, calo (bools; calo = accepted with neither of the
    two others), crv_mask, trk_mask (bit k set when gen primary k's descendants made the term), mu_mask (bit k set
    when gen primary k is a muon)."""
    import uproot
    fc, ft = uproot.open(crv_path), uproot.open(trk_path)
    gen = fc["dump/gen"].arrays(["subrun", "event", "idx", "pdg"], library="np")
    gkey = event_key(gen["subrun"], gen["event"])
    if gen["idx"].max(initial=0) > MAX_PRIMARIES:
        raise ValueError("more primaries in an event than the bit masks hold")
    events = np.unique(gkey)
    mu_mask = np.zeros(len(events), np.int64)
    gi = np.searchsorted(events, gkey)
    np.bitwise_or.at(mu_mask, gi, np.where(np.abs(gen["pdg"]) == cp.MUON, np.int64(1) << gen["idx"].astype(np.int64), 0))

    def term(steps, parts, edep_col, pcols, cfg):
        k = event_key(steps["subrun"], steps["event"])
        p = np.sqrt(sum(steps[c].astype(float) ** 2 for c in pcols))
        ok = (steps[edep_col] > cfg["min_edep"]) & (p > cfg["min_p"]) & (p < cfg["max_p"])
        e_acc, sp_acc = _accepting(k, steps["sp"].astype(np.int64), ok, cfg["min_steps"])
        pk = event_key(parts["subrun"], parts["event"])
        prim = _primary_of(pk, parts["key"].astype(np.int64), parts["parent"].astype(np.int64),
                           parts["genidx"].astype(np.int64))
        pid = particle_id(pk, parts["key"])
        o = np.argsort(pid)
        want = particle_id(e_acc, sp_acc)
        pos = np.searchsorted(pid[o], want)
        hit = (pos < len(pid)) & (pid[o][np.minimum(pos, len(pid) - 1)] == want)
        g = np.where(hit, prim[o][np.minimum(pos, len(pid) - 1)], -1)
        flag = np.zeros(len(events), bool)
        mask = np.zeros(len(events), np.int64)
        ei = np.searchsorted(events, e_acc)
        good = (ei < len(events)) & (events[np.minimum(ei, len(events) - 1)] == e_acc)
        flag[ei[good]] = True
        bits = np.where(g >= 0, np.int64(1) << np.clip(g, 0, MAX_PRIMARIES).astype(np.int64), 0)
        np.bitwise_or.at(mask, ei[good], bits[good])
        return flag, mask

    crv, crv_mask = term(fc["dump/steps"].arrays(["subrun", "event", "sp", "edep", "px0", "py0", "pz0"], library="np"),
                         fc["dump/particles"].arrays(["subrun", "event", "key", "parent", "genidx"], library="np"),
                         "edep", ("px0", "py0", "pz0"), CRV)
    trk, trk_mask = term(ft["dump/sgsteps"].arrays(["subrun", "event", "sp", "eion", "px", "py", "pz"], library="np"),
                         ft["dump/particles"].arrays(["subrun", "event", "key", "parent", "genidx"], library="np"),
                         "eion", ("px", "py", "pz"), TRK)
    return {"event": events, "crv": crv, "trk": trk, "calo": ~(crv | trk), "crv_mask": crv_mask,
            "trk_mask": trk_mask, "mu_mask": mu_mask}


# ----------------------------------------------------------------------------------------------- timing
def timing(db, gen_listing, dts_listing, skip_first=20):
    """Per generated event of a timed stock job: Geant4 time [s] (TimeTracker g4run), accepted flag (event in the
    dts listing). The first `skip_first` events (initialisation and warm-up) are dropped."""
    with sqlite3.connect(db) as con:
        rows = con.execute("select SubRun, Event, sum(Time) from TimeModule where ModuleLabel = 'g4run' "
                           "group by SubRun, Event").fetchall()
    a = np.array(rows, float).reshape(-1, 3)
    key = event_key(a[:, 0].astype(np.int64), a[:, 1].astype(np.int64))
    o = np.argsort(key)
    key, t = key[o], a[o, 2]
    key, t = key[skip_first:], t[skip_first:]
    _, gen_events = read_printgen(gen_listing)
    _, acc_events = read_printgen(dts_listing)
    if not np.isin(key, gen_events).all():
        raise ValueError("timed events missing from the gen listing")
    return {"event": key, "g4_s": t, "accepted": np.isin(key, acc_events)}


# ----------------------------------------------------------------------------------------------- curve
def param_grid(margins=DEFAULT_MARGINS):
    """The scanned parameter sets (cosmic_prefilter.Params)."""
    terms_sets = (cp.TERMS, ("crv",), ("trk", "calo"))
    emins = (None, 0.0, 10.0, 50.0)
    nonmuon = [("always", 0.0, None)] + [("threshold", e, nm) for e in (0.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
                                         for nm in (None, 1000.0, 4000.0) if not (e == 0.0 and nm is None)]
    out = []
    for terms, m, emin, (pol, nke, nmar) in itertools.product(terms_sets, margins, emins, nonmuon):
        if nmar is not None and nmar not in margins:
            continue
        out.append(cp.Params(margin_mm=m, min_muon_ke_mev=emin, nonmuon=pol, nonmuon_min_ke_mev=nke,
                             nonmuon_margin_mm=nmar, terms=terms))
    return out


class _Sample:
    """Features of one sample with the event bookkeeping every parameter set reuses."""

    def __init__(self, f, idx):
        self.f = f
        self.events, self.inv = np.unique(f.event, return_inverse=True)
        self.bits = np.int64(1) << np.asarray(idx, np.int64)
        self.mu = np.abs(f.pdg) == cp.MUON

    def keep_events(self, keep):
        return np.bincount(self.inv, weights=keep, minlength=len(self.events)) > 0

    def kept_bits(self, keep):
        out = np.zeros(len(self.events), np.int64)
        np.bitwise_or.at(out, self.inv[keep], self.bits[keep])
        return out

    def locate(self, keys):
        pos = np.minimum(np.searchsorted(self.events, keys), len(self.events) - 1)
        if not np.array_equal(self.events[pos], keys):
            raise ValueError("events missing from the features")
        return pos


def evaluate(ref, cen, params, tsample=None, tim=None):
    """Loss and saving of one parameter set. ref: _Sample of the reference primaries, cen: merged census (sorted by
    event, with `pos` = its rows in ref.events); tsample/tim: the timed sample (optional)."""
    keep = cp.decide_primaries(ref.f, params)
    kev = ref.keep_events(keep)
    lost = ~kev[cen["pos"]]
    n_acc = len(lost)
    loss = float(lost.mean()) if n_acc else None
    res = {"params": params.record(), "generated": int(len(kev)), "simulated": int(kev.sum()),
           "simulated_fraction": float(kev.mean()), "skipped_fraction": float(1 - kev.mean()),
           "accepted": int(n_acc), "lost": int(lost.sum()), "loss": loss,
           "loss_err": float(np.sqrt(loss * (1 - loss) / n_acc)) if n_acc else None}
    for t in ("crv", "trk", "calo"):
        sel = cen[t]
        res[f"accepted_{t}"] = int(sel.sum())
        res[f"lost_{t}"] = int((lost & sel).sum())
        res[f"loss_{t}"] = float(lost[sel].mean()) if sel.any() else None
    # components: origin of the accepting particles, and for a muon origin whether the muon's line missed the
    # envelopes grown by the margin (near miss: scattering, secondaries, decays) or only failed the range check
    origin = cen["crv_mask"] | cen["trk_mask"]
    mu_origin = (origin & cen["mu_mask"]) != 0
    known = origin != 0
    geo = cp.decide_primaries(ref.f, cp.Params(margin_mm=params.margin_mm, min_muon_ke_mev=None,
                                               terms=params.terms)) & ref.mu
    mu_geo_ok = (origin & cen["mu_mask"] & ref.kept_bits(geo)[cen["pos"]]) != 0
    comp = {"nonmuon": lost & known & ~mu_origin, "muon_near_miss": lost & mu_origin & ~mu_geo_ok,
            "muon_range": lost & mu_origin & mu_geo_ok, "calo_unknown_origin": lost & ~known}
    for k, v in comp.items():
        res[f"lost_{k}"] = int(v.sum())
    if tsample is not None:
        k = tsample.keep_events(cp.decide_primaries(tsample.f, params))[tim["pos"]]
        a = tim["accepted"]
        t_all = tim["g4_s"].sum()
        res["timing_events"] = int(len(k))
        res["timing_accepted"] = int(a.sum())
        res["g4_saved_fraction"] = float(tim["g4_s"][~k].sum() / t_all)
        res["timing_loss"] = float((a & ~k).sum() / max(a.sum(), 1))
        res["g4_s_per_accepted_stock"] = float(t_all / max(a.sum(), 1))
        res["g4_s_per_accepted_kept"] = float(tim["g4_s"][k].sum() / max((a & k).sum(), 1))
    return res


def _shard_index(feat_path):
    z = np.load(feat_path, allow_pickle=False)
    return z["idx"] if "idx" in z else None


def _merge(paths):
    parts = [dict(np.load(p, allow_pickle=False)) for p in paths]
    out = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    o = np.argsort(out["event"], kind="stable")
    out = {k: v[o] for k, v in out.items()}
    if len(np.unique(out["event"])) != len(out["event"]):
        raise ValueError("duplicate events in the inputs")
    return out


def _sample(paths):
    f = cp.Features.concat([cp.Features.load(p) for p in paths])
    return _Sample(f, np.concatenate([_shard_index(p) for p in paths]))


def curve(features, censuses, timing_features=(), timings=(), grid=None):
    ref = _sample(features)
    cen = _merge(censuses)
    cen["pos"] = ref.locate(cen["event"])
    tsample = tim = None
    if timing_features:
        tsample = _sample(timing_features)
        tim = _merge(timings)
        tim["pos"] = tsample.locate(tim["event"])
    grid = grid if grid is not None else param_grid(tuple(float(m) for m in ref.f.margins))
    return [evaluate(ref, cen, p, tsample, tim) for p in grid], ref.f


def pareto(rows, loss_key="loss", gain_key="g4_saved_fraction"):
    """Rows not dominated in (lower loss, higher saving)."""
    rows = [r for r in rows if r.get(gain_key) is not None]
    rows = sorted(rows, key=lambda r: (r[loss_key], -r[gain_key]))
    out, best = [], -np.inf
    for r in rows:
        if r[gain_key] > best:
            out.append(r)
            best = r[gain_key]
    return out


# ----------------------------------------------------------------------------------------------- CLI
def _primaries(path):
    if str(path).endswith((".txt", ".txt.gz")):
        return read_printgen(path)[0]
    return read_gen_table(path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mu2efastsim.cosmic_prefilter_scan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("features")
    a.add_argument("gen", nargs="+")
    a.add_argument("--gdml", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    a.add_argument("--margins", default=",".join(str(m) for m in DEFAULT_MARGINS))
    a = sub.add_parser("census")
    a.add_argument("--crv", required=True)
    a.add_argument("--trk", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("timing")
    a.add_argument("--db", required=True)
    a.add_argument("--gen", required=True)
    a.add_argument("--dts", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--skip-first", type=int, default=20)
    a = sub.add_parser("curve")
    a.add_argument("--features", nargs="+", required=True)
    a.add_argument("--census", nargs="+", required=True)
    a.add_argument("--timing-features", nargs="*", default=())
    a.add_argument("--timing", nargs="*", default=())
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "features":
        pf = cp.Prefilter.from_gdml(args.gdml, dtype=np.dtype(args.dtype))
        margins = [float(x) for x in args.margins.split(",")]
        feats, idxs, secs, n = [], [], 0.0, 0
        for g in args.gen:
            prim = _primaries(g)
            t0 = time.perf_counter()
            feats.append(pf.features(prim, margins))
            secs += time.perf_counter() - t0
            n += len(prim["pdg"])
            idxs.append(np.asarray(prim["idx"], np.int32))
        f = cp.Features.concat(feats)
        np.savez_compressed(args.out, event=f.event, pdg=f.pdg, tk=f.tk, margin_star=f.margin_star,
                            margins=f.margins, tk_env=f.tk_env, dtype=np.array(f.dtype),
                            unknown_materials=np.array(f.unknown_materials, dtype=str), idx=np.concatenate(idxs),
                            seconds=np.array(secs), primaries=np.array(n))
        print(json.dumps({"primaries": n, "seconds": secs, "primaries_per_s": n / max(secs, 1e-9),
                          "dtype": args.dtype, "unknown_materials": f.unknown_materials}))
    elif args.cmd == "census":
        c = census(args.crv, args.trk)
        np.savez_compressed(args.out, **c)
        print(json.dumps({"accepted": int(len(c["event"])), "crv": int(c["crv"].sum()), "trk": int(c["trk"].sum()),
                          "calo": int(c["calo"].sum())}))
    elif args.cmd == "timing":
        t = timing(args.db, args.gen, args.dts, args.skip_first)
        np.savez_compressed(args.out, **t)
        print(json.dumps({"events": int(len(t["event"])), "accepted": int(t["accepted"].sum()),
                          "g4_s": float(t["g4_s"].sum())}))
    else:
        rows, f = curve(args.features, args.census, args.timing_features, args.timing)
        out = {"dtype": f.dtype, "margins": f.margins.tolist(), "unknown_materials": f.unknown_materials,
               "rows": rows, "pareto": pareto(rows) if args.timing else pareto(rows, gain_key="skipped_fraction")}
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"{len(rows)} parameter sets -> {args.out}")


if __name__ == "__main__":
    main()
